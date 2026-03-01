"""話者ダイアリゼーションの動作確認・評価スクリプト。

高速化ポイント:
  1. voiceprint を全ファイル分まとめて事前ロード
  2. 音声読込+VAD (CPU) と embedding (GPU) をパイプラインで重畳
  3. DER 計算を ThreadPoolExecutor で並列化
"""

import glob
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from natsort import natsorted
from tqdm import tqdm

from utils import SpeakerDiarizer, compute_der

_log_fmt = "%(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_log_fmt, handlers=[
    logging.StreamHandler(),
    logging.FileHandler("eval.log", mode="w", encoding="utf-8"),
])


def get_audio_files(data_dir: str = "./input", split: str = "train") -> list[str]:
    return natsorted(glob.glob(os.path.join(data_dir, split, "**", "*.wav")))


def _labels_by_audio_id(df_label: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """audio_id ごとにラベル DataFrame を分割し辞書で返す（ループ内の filter を回避）。"""
    parts = df_label.partition_by("audio_id", as_dict=True)
    return {key[0]: part for key, part in parts.items()}


def _compute_der_task(
    audio_id: str,
    df_infer: pl.DataFrame,
    df_label: pl.DataFrame,
) -> pl.DataFrame:
    """DER 計算タスク (ThreadPoolExecutor 向け)。"""
    df_der = compute_der(df_infer=df_infer, df_label=df_label)
    return df_der.with_columns(
        pl.lit(audio_id).alias("audio_id"),
        pl.lit("train").alias("train_test"),
    )


def evaluate(
    diarizer: SpeakerDiarizer,
    audio_files: list[str],
    df_label: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """パイプライン方式で推論 + DER 評価を行う。

    CPU (音声読込+VAD)  →  GPU (embedding+マッチング)  →  CPU (DER)
    を重畳して実行し、GPU の空き時間を削減する。
    """
    dfs_infer: list[pl.DataFrame] = []
    der_futures: list[tuple[str, Future]] = []
    label_by_audio = _labels_by_audio_id(df_label)
    logger = logging.getLogger(__name__)

    n = len(audio_files)
    if n == 0:
        empty = pl.DataFrame()
        return empty, empty

    with ThreadPoolExecutor(max_workers=4) as pool:
        # --- プリフェッチ: 最初のファイルの VAD を先行実行 ---
        next_prep_future: Future = pool.submit(diarizer.prepare_vad, audio_files[0])

        for i in tqdm(range(n)):
            # (1) CPU ステージの結果を取得（既にバックグラウンドで完了済みのはず）
            prepared = next_prep_future.result()

            # (2) 次のファイルの CPU ステージをバックグラウンドで開始
            if i + 1 < n:
                next_prep_future = pool.submit(diarizer.prepare_vad, audio_files[i + 1])

            # (3) GPU ステージ: embedding + マッチング（メインスレッド）
            df_infer_tmp = diarizer.diarize_from_prepared(prepared)
            dfs_infer.append(df_infer_tmp)

            # (4) DER 計算をバックグラウンドへ投入
            audio_id = prepared.audio_id
            df_label_tmp = label_by_audio.get(audio_id, pl.DataFrame())
            future = pool.submit(_compute_der_task, audio_id, df_infer_tmp, df_label_tmp)
            der_futures.append((audio_id, future))

    # --- DER 結果を完了次第ログ出力して回収 ---
    dfs_der: list[pl.DataFrame] = []
    for future in as_completed(f for _, f in der_futures):
        df_der_tmp = future.result()
        row = df_der_tmp.row(0, named=True)
        audio_id = row["audio_id"]
        logger.info(
            "[%s] DER=%.4f  (confusion=%.4f  false_alarm=%.4f  missed=%.4f)",
            audio_id, row["DER"], row["confusion"],
            row["false_alarm"], row["missed_detection"],
        )
        dfs_der.append(df_der_tmp)

    return pl.concat(dfs_infer), pl.concat(dfs_der)


def main() -> None:
    train_audio_files = get_audio_files()
    print(f"train データ: {len(train_audio_files)} ファイル")

    diarizer = SpeakerDiarizer()
    df_label = pl.read_csv(os.path.join("./input", "train_annotation.csv"))

    # voiceprint を全ディレクトリ分まとめて事前ロード
    audio_dirs = list({str(Path(f).parent) for f in train_audio_files})
    diarizer.preload_all_voiceprints(audio_dirs)

    df_infer, df_der = evaluate(diarizer, train_audio_files, df_label)

    print(f"\nmean DER: {df_der['DER'].mean():.4f}")
    print(df_der.select("audio_id", "DER", "confusion", "false_alarm", "missed_detection"))


if __name__ == "__main__":
    main()
