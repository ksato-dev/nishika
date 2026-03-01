import os
import glob
import logging
from pathlib import Path

import torch
import numpy as np
import polars as pl
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Inference, Model
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from scipy.spatial.distance import cdist

from config import (
    AUDIO_POWER_NORMALIZE,
    EMBED_BATCH_SIZE,
    VAD_MIN_DURATION_FILTER,
    VAD_MIN_DURATION_SEC,
    REMATCH_ENABLED,
    REMATCH_SIMILARITY_THRESHOLD,
)

import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_EMB_BATCH_SIZE = EMBED_BATCH_SIZE


def _power_normalize(waveform: torch.Tensor) -> torch.Tensor:
    """RMS パワー正規化。録音レベル差を吸収する。"""
    rms = waveform.square().mean().sqrt()
    return waveform / (rms + 1e-8)


class PreparedData:
    """CPU ステージ (音声読込+VAD) の結果を保持する軽量コンテナ。"""
    __slots__ = (
        "audio_path", "audio_id", "train_test", "audio_dir",
        "audio_tensor", "seg_pairs", "raw_ends",
    )

    def __init__(
        self,
        audio_path: str,
        audio_id: str,
        train_test: str,
        audio_dir: str,
        audio_tensor: torch.Tensor | None,
        seg_pairs: list[tuple[float, float]],
        raw_ends: list[float],
    ):
        self.audio_path = audio_path
        self.audio_id = audio_id
        self.train_test = train_test
        self.audio_dir = audio_dir
        self.audio_tensor = audio_tensor
        self.seg_pairs = seg_pairs
        self.raw_ends = raw_ends

_HAS_ONNXRUNTIME = False
try:
    import onnxruntime  # noqa: F401
    _HAS_ONNXRUNTIME = True
except ImportError:
    pass


class SpeakerDiarizer:
    """話者ダイアリゼーション用モデルをまとめて管理するクラス。

    モデルを一度だけロードし、複数の音声ファイルに対して再利用できる。
    """

    def __init__(self, hf_token: str | None = None, embed_batch_size: int = _EMB_BATCH_SIZE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._voiceprints_cache: dict[str, dict[str, np.ndarray]] = {}
        self._vp_matrix_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self.embed_batch_size = embed_batch_size
        self._load_models(hf_token or os.getenv("HF_ACCESS_TOKEN"))

    def _load_models(self, hf_token: str | None) -> None:
        logger.info("モデルをロード中...")

        use_onnx = _HAS_ONNXRUNTIME
        self.silero_vad = load_silero_vad(onnx=use_onnx)
        if use_onnx:
            logger.info("Silero VAD: ONNX バックエンド")
        else:
            logger.info("Silero VAD: JIT バックエンド (onnxruntime をインストールすると ~3 倍高速化)")

        pyannote_emb = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=hf_token
        )
        pyannote_emb.to(self.device)
        self.inference = Inference(pyannote_emb, window="whole")
        logger.info("モデルロード完了")

    def _detect_speech_from_tensor(self, audio_tensor: torch.Tensor) -> list[dict]:
        """メモリ上の音声テンソルから Silero VAD で発話区間を検出する。"""
        return get_speech_timestamps(
            audio_tensor, self.silero_vad, return_seconds=True
        )

    def _inference_normalized(self, wav_path: str) -> np.ndarray:
        """音声ファイルを正規化してから embedding を計算する。"""
        if AUDIO_POWER_NORMALIZE:
            waveform = read_audio(wav_path)
            waveform = _power_normalize(waveform)
            audio_dict = {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": SAMPLE_RATE,
            }
            return self.inference(audio_dict)
        return self.inference(wav_path)

    def load_voiceprints(self, audio_dir: str) -> dict[str, np.ndarray]:
        """声紋ファイルを読み込み、ラベル→埋め込みの辞書を返す。同一ディレクトリはキャッシュする。

        各 voiceprint wav に VAD をかけて発話区間を分割し、区間ごとの embedding を計算する。
        同一話者の複数区間は同じラベルで ``_vp_matrix_cache`` に格納され、
        マッチング時に最も類似度の高い区間が選ばれる。
        """
        if audio_dir in self._voiceprints_cache:
            return self._voiceprints_cache[audio_dir]
        vp_paths = sorted(
            glob.glob(os.path.join(audio_dir, "voiceprints", "[A-Z].wav"))
        )

        result: dict[str, np.ndarray] = {}
        ref_labels: list[str] = []
        ref_embeds: list[np.ndarray] = []

        for p in vp_paths:
            speaker = Path(p).stem
            audio_tensor = read_audio(p)
            if AUDIO_POWER_NORMALIZE:
                audio_tensor = _power_normalize(audio_tensor)

            audio_length = len(audio_tensor) / SAMPLE_RATE
            vad_segments = self._detect_speech_from_tensor(audio_tensor)
            seg_pairs: list[tuple[float, float]] = []
            for seg in vad_segments:
                start, end = seg["start"], seg["end"]
                if start >= audio_length:
                    continue
                duration = min(end, audio_length) - start
                if VAD_MIN_DURATION_FILTER and duration < VAD_MIN_DURATION_SEC:
                    continue
                seg_pairs.append((start, min(end, audio_length)))

            if seg_pairs:
                embeds = self._batch_embed(audio_tensor, seg_pairs)
                for emb in embeds:
                    ref_labels.append(speaker)
                    ref_embeds.append(emb.reshape(1, -1))
                result[speaker] = embeds[0]
                logger.info("voiceprint %s: VAD %d 区間 → %d embeddings", speaker, len(seg_pairs), len(embeds))
            else:
                emb = self._inference_normalized(p)
                ref_labels.append(speaker)
                ref_embeds.append(emb.reshape(1, -1))
                result[speaker] = emb
                logger.info("voiceprint %s: VAD 区間なし → 全体 1 embedding", speaker)

        self._voiceprints_cache[audio_dir] = result
        if ref_embeds:
            self._vp_matrix_cache[audio_dir] = (
                ref_labels,
                np.vstack(ref_embeds),
            )
        return result

    def preload_all_voiceprints(self, audio_dirs: list[str]) -> None:
        """複数ディレクトリの voiceprint を一括ロードしてキャッシュに載せる。"""
        for d in audio_dirs:
            self.load_voiceprints(d)

    # ------------------------------------------------------------------
    # CPU ステージ: 音声読込 + VAD (GPU 不要 → 別スレッドで実行可)
    # ------------------------------------------------------------------
    def prepare_vad(self, audio_path: str) -> PreparedData:
        """音声読込と VAD を実行し、GPU 推論に必要なデータを返す。"""
        audio_id = Path(audio_path).stem
        train_test = "train" if "train" in audio_path else "test"
        audio_dir = str(Path(audio_path).parent)

        audio_tensor = read_audio(audio_path)
        if AUDIO_POWER_NORMALIZE:
            audio_tensor = _power_normalize(audio_tensor)
        audio_length = len(audio_tensor) / SAMPLE_RATE
        vad_result = self._detect_speech_from_tensor(audio_tensor)

        seg_pairs: list[tuple[float, float]] = []
        raw_ends: list[float] = []
        for seg in vad_result:
            start, end = seg["start"], seg["end"]
            if start >= audio_length:
                continue
            duration = min(end, audio_length) - start
            if VAD_MIN_DURATION_FILTER and duration < VAD_MIN_DURATION_SEC:
                continue
            seg_pairs.append((start, min(end, audio_length)))
            raw_ends.append(end)

        return PreparedData(
            audio_path=audio_path,
            audio_id=audio_id,
            train_test=train_test,
            audio_dir=audio_dir,
            audio_tensor=audio_tensor if seg_pairs else None,
            seg_pairs=seg_pairs,
            raw_ends=raw_ends,
        )

    # ------------------------------------------------------------------
    # GPU ステージ: embedding + マッチング
    # ------------------------------------------------------------------
    def diarize_from_prepared(self, prep: PreparedData) -> pl.DataFrame:
        """prepare_vad の結果から embedding 計算→話者マッチングを実行する。"""
        voiceprints = self.load_voiceprints(prep.audio_dir)

        if not voiceprints or not prep.seg_pairs or prep.audio_tensor is None:
            prep.audio_tensor = None
            return self._empty_result(prep.audio_id, prep.train_test)

        try:
            embeddings = self._batch_embed(prep.audio_tensor, prep.seg_pairs)
            prep.audio_tensor = None  # embedding 計算後は不要 → 即解放

            matched_labels, cos_sims = self._match_speakers_batch(embeddings, prep.audio_dir)

            if REMATCH_ENABLED:
                matched_labels, cos_sims = self._rematch_with_enriched_voiceprints(
                    embeddings, matched_labels, cos_sims, prep.audio_dir,
                )
        except Exception as e:
            logger.warning("バッチ処理エラー (%s): %s — フォールバック", prep.audio_id, e)
            result = self._diarize_sequential(
                prep.audio_id, prep.train_test, prep.audio_tensor,
                prep.audio_dir, prep.seg_pairs, prep.raw_ends, voiceprints,
            )
            prep.audio_tensor = None
            return result

        valid_starts = [s for s, _ in prep.seg_pairs]
        df = pl.DataFrame({
            "audio_id": [prep.audio_id] * len(valid_starts),
            "train_test": [prep.train_test] * len(valid_starts),
            "start_time": valid_starts,
            "end_time": prep.raw_ends,
            "target": matched_labels,
            "cos_sim": cos_sims.tolist(),
        })
        return df.select(["audio_id", "train_test", "start_time", "end_time", "target", "cos_sim"])

    # ------------------------------------------------------------------
    # バッチ埋め込み
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _batch_embed(
        self,
        audio_tensor: torch.Tensor,
        segments: list[tuple[float, float]],
    ) -> np.ndarray:
        """VAD セグメント群の埋め込みを一括計算する。

        長さの近いセグメントをまとめてパディング → Inference.infer で
        バッチ forward pass を実行し、N 回→ ceil(N/batch) 回に削減する。

        Returns:
            (N, embed_dim) の numpy 配列
        """
        n = len(segments)
        if n == 0:
            return np.empty((0, 0))

        lengths: list[int] = []
        sample_ranges: list[tuple[int, int]] = []
        for start, end in segments:
            s = int(start * SAMPLE_RATE)
            e = int(end * SAMPLE_RATE)
            sample_ranges.append((s, e))
            lengths.append(e - s)

        order = sorted(range(n), key=lambda i: lengths[i])

        all_embeds: list[np.ndarray] = [None] * n  # type: ignore[list-item]
        bs = self.embed_batch_size
        for bi in range(0, n, bs):
            idx_batch = order[bi : bi + bs]
            max_len = max(lengths[i] for i in idx_batch)
            batch_tensor = torch.zeros(len(idx_batch), 1, max_len)
            for j, idx in enumerate(idx_batch):
                s, e = sample_ranges[idx]
                batch_tensor[j, 0, : lengths[idx]] = audio_tensor[s:e]
            embeds = self.inference.infer(batch_tensor)
            for j, idx in enumerate(idx_batch):
                all_embeds[idx] = embeds[j]
            del batch_tensor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.vstack(all_embeds)

    # ------------------------------------------------------------------
    # 一括話者マッチング
    # ------------------------------------------------------------------
    def _match_speakers_batch(
        self,
        embeddings: np.ndarray,
        audio_dir: str,
    ) -> tuple[list[str], np.ndarray]:
        """全セグメントの埋め込みと声紋を一括比較する。

        Returns:
            (labels_per_segment, cos_sim_per_segment)
        """
        if audio_dir not in self._vp_matrix_cache:
            return [], np.array([])
        labels, vp_matrix = self._vp_matrix_cache[audio_dir]
        cos_sims = 1.0 - cdist(embeddings, vp_matrix, metric="cosine")  # (N, K)
        best_idx = np.argmax(cos_sims, axis=1)
        best_sim = cos_sims[np.arange(len(best_idx)), best_idx]
        matched_labels = [labels[i] for i in best_idx]
        return matched_labels, best_sim

    # ------------------------------------------------------------------
    # 再検索: voiceprint 強化 → 再マッチング
    # ------------------------------------------------------------------
    def _rematch_with_enriched_voiceprints(
        self,
        embeddings: np.ndarray,
        first_labels: list[str],
        first_sims: np.ndarray,
        audio_dir: str,
        threshold: float = REMATCH_SIMILARITY_THRESHOLD,
    ) -> tuple[list[str], np.ndarray]:
        """高類似度セグメントを参照セットとし、未マッチセグメントを最近傍で再分類する。

        1回目で cos_sim >= threshold のセグメントを「確定」とみなし、
        voiceprint + 確定セグメントの embedding を参照セットとして構築。
        未確定セグメントを参照セットの個々の embedding と 1対1 で比較し、
        最も類似度の高い参照の話者ラベルを割り当てる。

        Returns:
            (labels_per_segment, cos_sim_per_segment)
        """
        if audio_dir not in self._vp_matrix_cache:
            return first_labels, first_sims
        orig_labels, orig_vp_matrix = self._vp_matrix_cache[audio_dir]

        confirmed_mask = np.array([sim >= threshold for sim in first_sims], dtype=bool)
        unconfirmed_idx = np.where(~confirmed_mask)[0]

        if len(unconfirmed_idx) == 0:
            return first_labels, first_sims

        # 参照セット: voiceprint + 確定セグメント (個々の embedding をそのまま保持)
        ref_embeds: list[np.ndarray] = []
        ref_labels: list[str] = []
        for k, speaker in enumerate(orig_labels):
            ref_embeds.append(orig_vp_matrix[k].reshape(1, -1))
            ref_labels.append(speaker)
        for i in np.where(confirmed_mask)[0]:
            ref_embeds.append(embeddings[i].reshape(1, -1))
            ref_labels.append(first_labels[i])

        ref_matrix = np.vstack(ref_embeds)  # (R, embed_dim)

        # 未確定セグメントを参照セットの全 embedding と 1対1 比較
        unconfirmed_embeds = embeddings[unconfirmed_idx]  # (U, embed_dim)
        cos_sims = 1.0 - cdist(unconfirmed_embeds, ref_matrix, metric="cosine")  # (U, R)
        best_ref_idx = np.argmax(cos_sims, axis=1)
        best_ref_sim = cos_sims[np.arange(len(best_ref_idx)), best_ref_idx]

        # 結果を組み立て（再検索の類似度が1回目より低い場合は上書きしない）
        final_labels = list(first_labels)
        final_sims = np.array(first_sims, dtype=np.float64)
        n_changed = 0
        for j, orig_i in enumerate(unconfirmed_idx):
            first_sim = first_sims[orig_i]
            rematch_sim = best_ref_sim[j]
            if rematch_sim < first_sim:
                continue
            new_label = ref_labels[best_ref_idx[j]]
            if final_labels[orig_i] != new_label:
                n_changed += 1
            final_labels[orig_i] = new_label
            final_sims[orig_i] = rematch_sim

        logger.info(
            "再検索: 確定=%d 未確定=%d → %d 件ラベル変更 (threshold=%.2f)",
            int(confirmed_mask.sum()), len(unconfirmed_idx), n_changed, threshold,
        )

        return final_labels, final_sims

    # ------------------------------------------------------------------
    def diarize(self, audio_path: str) -> pl.DataFrame:
        """音声ファイルに対して話者ダイアリゼーションを実行する（一括版）。

        Returns:
            カラム: audio_id, train_test, start_time, end_time, target, cos_sim
        """
        return self.diarize_from_prepared(self.prepare_vad(audio_path))

    # 逐次フォールバック
    def _diarize_sequential(
        self,
        audio_id: str,
        train_test: str,
        audio_tensor: torch.Tensor,
        audio_dir: str,
        seg_pairs: list[tuple[float, float]],
        raw_ends: list[float],
        voiceprints: dict[str, np.ndarray],
    ) -> pl.DataFrame:
        audio_inmem: dict = {
            "waveform": audio_tensor.unsqueeze(0),
            "sample_rate": SAMPLE_RATE,
        }
        results: list[dict] = []
        for (start, end), raw_end in zip(seg_pairs, raw_ends):
            try:
                embed = self.inference.crop(audio_inmem, Segment(start, end))
                labels, vp_matrix = self._vp_matrix_cache[audio_dir]
                seg = embed.reshape(1, -1)
                cos_sims = 1.0 - cdist(seg, vp_matrix, metric="cosine")[0]
                idx = int(np.argmax(cos_sims))
                results.append({
                    "start_time": start, "end_time": raw_end,
                    "target": labels[idx], "cos_sim": float(cos_sims[idx]),
                })
            except Exception as e:
                logger.warning("セグメント処理エラー (%s): %s", audio_id, e)
        if not results:
            return self._empty_result(audio_id, train_test)
        df = pl.DataFrame(results)
        return df.with_columns(
            pl.lit(audio_id).alias("audio_id"),
            pl.lit(train_test).alias("train_test"),
        ).select(["audio_id", "train_test", "start_time", "end_time", "target", "cos_sim"])

    @staticmethod
    def _empty_result(audio_id: str, train_test: str) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "audio_id": pl.Utf8, "train_test": pl.Utf8,
                "start_time": pl.Float64, "end_time": pl.Float64,
                "target": pl.Utf8, "cos_sim": pl.Float64,
            }
        )


def _df_to_annotation(df: pl.DataFrame, time_cols: tuple[str, str], label_col: str) -> Annotation:
    """DataFrame → pyannote Annotation への高速変換（カラム直接アクセス）。"""
    ann = Annotation()
    if df.is_empty():
        return ann
    starts = df[time_cols[0]].to_list()
    ends = df[time_cols[1]].to_list()
    labels = df[label_col].to_list()
    for s, e, lbl in zip(starts, ends, labels):
        ann[Segment(s, e)] = lbl
    return ann


def compute_der(
    df_label: pl.DataFrame,
    df_infer: pl.DataFrame,
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> pl.DataFrame:
    """正解データと推論データの間の DER を計算する。

    Args:
        df_label: 正解ラベル (start_time, end_time, speaker)
        df_infer: 推論結果 (start_time, end_time, target)
        collar: 発話境界の許容誤差（秒）
        skip_overlap: 重複発話区間を除外するか
    """
    label = _df_to_annotation(df_label, ("start_time", "end_time"), "speaker")
    hypothesis = _df_to_annotation(df_infer, ("start_time", "end_time"), "target")

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    der = metric(label, hypothesis)
    components = metric(label, hypothesis, detailed=True)
    total = components["total"]
    if total == 0:
        total = 1.0

    return pl.DataFrame({
        "DER": der,
        "confusion": components["confusion"] / total,
        "false_alarm": components["false alarm"] / total,
        "missed_detection": components["missed detection"] / total,
        "components": components,
    })
