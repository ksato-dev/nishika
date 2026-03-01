"""EDA（探索的データ分析）用スクリプト。"""

import glob
import os
from pathlib import Path

from config import (
    EDA_LOUDNESS,
    EDA_SEGMENT_LOUDNESS,
    EDA_VOICEPRINT_LOUDNESS,
    EDA_VAD_CLUSTERING,
    EDA_VAD_CLUSTERING_INPUT,
    EDA_VAD_CLUSTERING_METHOD,
    EDA_VAD_CLUSTERING_N_CLUSTERS,
    EDA_VAD_CLUSTERING_SIMILARITY_THRESHOLD,
    EDA_VAD_VOICEPRINT_MATCH,
    EDA_VAD_VOICEPRINT_MATCH_THRESHOLD,
    VAD_MIN_DURATION_FILTER,
    VAD_MIN_DURATION_SEC,
    SPECTROGRAM_DPI,
    SPECTROGRAM_FMAX,
    SPECTROGRAM_FOLDER,
    MEL_INPUT_PATH,
    SPECTRUM_INPUT_PATH,
    SPECTROGRAM_N_MELS,
    SPECTROGRAM_OUTPUT_DIR,
    SPECTROGRAM_SINGLE,
    SPECTROGRAM_SR,
    SPECTRUM_CENTER_SEC,
    SPECTRUM_SCAN_ALL,
    SPECTRUM_STEP_SEC,
    SPECTRUM_FREQ_MAX,
    SPECTRUM_FREQ_MIN,
    SPECTRUM_N_FFT,
    SPECTRUM_BANDPASS_ENABLED,
    SPECTRUM_BANDPASS_KEEP_HIGH,
    SPECTRUM_BANDPASS_KEEP_LOW,
    SPECTRUM_MA_ENABLED,
    SPECTRUM_MA_WINDOW,
    SPECTRUM_SMOOTH_ENABLED,
    SPECTRUM_SMOOTH_POLYORDER,
    SPECTRUM_SMOOTH_WINDOW,
    SPECTRUM_WINDOW_SEC,
    DECAY_FIT_ENABLED,
    DECAY_FIT_FREQ_MAX,
    LPC_BW_MAX,
    LPC_BW_MIN,
    LPC_ENABLED,
    LPC_FORMANT_MAX_FREQ,
    LPC_FORMANT_MIN_FREQ,
    LPC_MAX_FORMANTS,
    LPC_MIN_ENERGY_DB,
    LPC_ORDER,
    LPC_PREEMPHASIS,
)

import librosa
import matplotlib.pyplot as plt
import polars as pl
import pyloudnorm as pyln
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = 16000
INPUT_DIR = "./input"
OUTPUT_DIR = "./eda"

# 高速読み込み: soundfile があれば利用（16kHz のときリサンプル不要）
try:
    import soundfile as sf
    _USE_SOUNDFILE = True
except ImportError:
    _USE_SOUNDFILE = False


def get_audio_files(split: str = "train") -> list[str]:
    return sorted(glob.glob(os.path.join(INPUT_DIR, split, "**", "*.wav")))


def _load_audio_fast(wav_path: str) -> tuple[list[float], int]:
    """1ファイルだけ読み込み。soundfile 利用時はリサンプルが必要なときのみ librosa。"""
    if _USE_SOUNDFILE:
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        return data.tolist(), SAMPLE_RATE
    waveform = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)[0]
    return waveform.tolist(), SAMPLE_RATE


def _measure_loudness(wav_path: str) -> float:
    """1ファイルの loudness (LUFS) を返す。"""
    meter = pyln.Meter(SAMPLE_RATE, block_size=0.1)
    waveform, _ = _load_audio_fast(wav_path)
    return meter.integrated_loudness(np.asarray(waveform, dtype=np.float64))


def compute_loudness(audio_files: list[str], split: str) -> pl.DataFrame:
    """各音声ファイルの loudness (LUFS) を計算して DataFrame で返す。"""
    records: list[dict] = []
    for wav_path in tqdm(audio_files, desc=f"loudness計算 ({split})"):
        records.append({
            "audio_id": Path(wav_path).stem,
            "split": split,
            "loudness": _measure_loudness(wav_path),
        })
    print(f"  → {split}: {len(records)}/{len(audio_files)} 件計算完了")
    return pl.DataFrame(records)


def compute_voiceprint_loudness(split: str = "train") -> pl.DataFrame:
    """voiceprints 内の各話者音声の loudness を計算して DataFrame で返す。"""
    vp_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, split, "*", "voiceprints", "[A-Z].wav"))
    )
    records: list[dict] = []
    for vp_path in tqdm(vp_files, desc=f"voiceprint loudness ({split})"):
        p = Path(vp_path)
        records.append({
            "audio_id": p.parents[1].name,
            "speaker": p.stem,
            "split": split,
            "loudness": _measure_loudness(vp_path),
        })
    print(f"  → {split} voiceprints: {len(records)}/{len(vp_files)} 件計算完了")
    return pl.DataFrame(records)


def compute_segment_loudness(audio_files: list[str], split: str) -> pl.DataFrame:
    """VAD で検知した各発話区間の loudness を計算して DataFrame で返す。"""
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

    vad_model = load_silero_vad()
    meter = pyln.Meter(SAMPLE_RATE, block_size=0.1)
    min_samples = int(0.1 * SAMPLE_RATE)
    records: list[dict] = []

    for wav_path in tqdm(audio_files, desc=f"segment loudness ({split})"):
        audio_id = Path(wav_path).stem
        audio_tensor = read_audio(wav_path)
        waveform_np = audio_tensor.numpy()
        segments = get_speech_timestamps(audio_tensor, vad_model, return_seconds=True)

        for seg in segments:
            start_sample = int(seg["start"] * SAMPLE_RATE)
            end_sample = int(seg["end"] * SAMPLE_RATE)
            chunk = waveform_np[start_sample:end_sample]
            if len(chunk) < min_samples:
                continue
            loudness = meter.integrated_loudness(chunk.astype(np.float64))
            records.append({
                "audio_id": audio_id,
                "split": split,
                "start_time": seg["start"],
                "end_time": seg["end"],
                "duration": seg["end"] - seg["start"],
                "loudness": loudness,
            })

    print(f"  → {split} segments: {len(records)} 区間")
    return pl.DataFrame(records)


def compute_vad_clustering(wav_path: str, output_dir: str = OUTPUT_DIR) -> pl.DataFrame:
    """単一音声ファイルに対し VAD → embedding → クラスタリングを行い結果を CSV 出力する。

    Returns:
        クラスタリング結果の DataFrame
    """
    import torch
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from scipy.spatial.distance import cdist

    wav_path = os.path.normpath(wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {wav_path}")

    audio_id = Path(wav_path).stem
    print(f"[VAD clustering] 対象: {wav_path}")

    # --- VAD ---
    vad_model = load_silero_vad()
    audio_tensor = read_audio(wav_path)
    segments = get_speech_timestamps(audio_tensor, vad_model, return_seconds=True)
    print(f"  VAD 検出区間: {len(segments)}")

    if not segments:
        print("  発話区間が見つかりません")
        return pl.DataFrame()

    # 最小区間フィルタ
    filtered_segments: list[dict] = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if VAD_MIN_DURATION_FILTER and duration < VAD_MIN_DURATION_SEC:
            continue
        filtered_segments.append(seg)

    skipped = len(segments) - len(filtered_segments)
    if skipped > 0:
        print(f"  最小区間フィルタ: {skipped} 区間を除外 (< {VAD_MIN_DURATION_SEC}s)")
    segments = filtered_segments

    if not segments:
        print("  フィルタ後に発話区間が残りません")
        return pl.DataFrame()

    # --- Embedding ---
    from utils import SpeakerDiarizer
    diarizer = SpeakerDiarizer()

    seg_pairs = [(seg["start"], seg["end"]) for seg in segments]
    embeddings = diarizer._batch_embed(audio_tensor, seg_pairs)
    print(f"  Embedding 計算完了: shape={embeddings.shape}")

    # --- クラスタリング ---
    n_segments = embeddings.shape[0]
    labels = np.zeros(n_segments, dtype=int)

    if EDA_VAD_CLUSTERING:
        n_clusters = EDA_VAD_CLUSTERING_N_CLUSTERS
        method = EDA_VAD_CLUSTERING_METHOD

        cos_sim = 1.0 - cdist(embeddings, embeddings, metric="cosine")

        triu_idx = np.triu_indices(n_segments, k=1)
        pairwise_sim = cos_sim[triu_idx]
        print(f"  コサイン類似度統計 (ペアワイズ {len(pairwise_sim)} 組):")
        print(f"    min={pairwise_sim.min():.4f}  Q1={np.percentile(pairwise_sim, 25):.4f}"
              f"  median={np.median(pairwise_sim):.4f}  Q3={np.percentile(pairwise_sim, 75):.4f}"
              f"  max={pairwise_sim.max():.4f}")
        print(f"  similarity_threshold={EDA_VAD_CLUSTERING_SIMILARITY_THRESHOLD}")

        distance_threshold = 1.0 - EDA_VAD_CLUSTERING_SIMILARITY_THRESHOLD

        if method == "agglomerative":
            cos_dist = 1.0 - cos_sim
            if n_clusters is not None:
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage="average",
                )
            else:
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    metric="precomputed",
                    linkage="average",
                )
            labels = model.fit_predict(cos_dist)
        elif method == "spectral":
            affinity = np.clip(cos_sim, 0, None)
            np.fill_diagonal(affinity, 1.0)
            k = n_clusters or min(n_segments, 8)
            model = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=42,
            )
            labels = model.fit_predict(affinity)
        else:
            raise ValueError(f"未対応のクラスタリング手法: {method}")

        n_found = len(set(labels))
        print(f"  クラスタリング完了: {n_found} グループ ({method})")

    # --- Voiceprint 照合 ---
    vp_speakers: list[str] = []
    vp_cos_sims: list[float] = []
    vp_matched: list[bool] = []

    if EDA_VAD_VOICEPRINT_MATCH:
        audio_dir = str(Path(wav_path).parent)
        voiceprints = diarizer.load_voiceprints(audio_dir)
        if voiceprints:
            matched_labels, best_sims = diarizer._match_speakers_batch(embeddings, audio_dir)
            threshold = EDA_VAD_VOICEPRINT_MATCH_THRESHOLD
            for lbl, sim in zip(matched_labels, best_sims):
                is_match = float(sim) >= threshold
                vp_speakers.append(lbl if is_match else "")
                vp_cos_sims.append(float(sim))
                vp_matched.append(is_match)
            n_matched = sum(vp_matched)
            print(f"  Voiceprint 照合: {n_matched}/{len(vp_matched)} 一致 "
                  f"(threshold={threshold})")
        else:
            print("  Voiceprint が見つかりません — 照合スキップ")

    # --- DataFrame 構築 ---
    records: list[dict] = []
    for i, seg in enumerate(segments):
        row: dict = {
            "audio_id": audio_id,
            "segment_idx": i,
            "start_time": seg["start"],
            "end_time": seg["end"],
            "duration": seg["end"] - seg["start"],
        }
        if EDA_VAD_CLUSTERING:
            row["cluster"] = int(labels[i])
        if vp_speakers:
            row["vp_speaker"] = vp_speakers[i]
            row["vp_cos_sim"] = vp_cos_sims[i]
            row["vp_matched"] = vp_matched[i]
        records.append(row)

    sort_keys = ["cluster", "start_time"] if EDA_VAD_CLUSTERING else ["start_time"]
    df = pl.DataFrame(records).sort(sort_keys)

    os.makedirs(output_dir, exist_ok=True)

    # クラスタ内訳サマリ（クラスタリングがオンの場合のみ）
    if EDA_VAD_CLUSTERING:
        agg_exprs = [
            pl.len().alias("n_segments"),
            pl.col("duration").sum().alias("total_duration"),
            pl.col("duration").mean().alias("mean_duration"),
            pl.col("start_time").min().alias("earliest_start"),
            pl.col("end_time").max().alias("latest_end"),
        ]
        if vp_speakers:
            agg_exprs.extend([
                pl.col("vp_matched").sum().alias("n_vp_matched"),
                pl.col("vp_cos_sim").mean().alias("mean_vp_cos_sim"),
            ])
        summary = df.group_by("cluster").agg(agg_exprs).sort("cluster")

        detail_path = os.path.join(output_dir, f"vad_cluster_detail_{audio_id}.csv")
        summary_path = os.path.join(output_dir, f"vad_cluster_summary_{audio_id}.csv")
        df.write_csv(detail_path)
        summary.write_csv(summary_path)
        print(f"  詳細 CSV: {detail_path} ({len(df)} 行)")
        print(f"  サマリ CSV: {summary_path} ({len(summary)} 行)")
        print(summary)

    # --- Voiceprint 照合の一致/不一致サマリ ---
    if vp_speakers:
        df_matched = df.filter(pl.col("vp_matched"))
        df_unmatched = df.filter(~pl.col("vp_matched"))

        match_summary_rows: list[dict] = []
        for speaker in sorted(set(s for s in vp_speakers if s)):
            sub = df_matched.filter(pl.col("vp_speaker") == speaker)
            if sub.is_empty():
                continue
            match_summary_rows.append({
                "group": speaker,
                "n_segments": len(sub),
                "total_duration": sub["duration"].sum(),
                "mean_duration": sub["duration"].mean(),
                "mean_vp_cos_sim": sub["vp_cos_sim"].mean(),
                "earliest_start": sub["start_time"].min(),
                "latest_end": sub["end_time"].max(),
            })
        if not df_unmatched.is_empty():
            match_summary_rows.append({
                "group": "(unmatched)",
                "n_segments": len(df_unmatched),
                "total_duration": df_unmatched["duration"].sum(),
                "mean_duration": df_unmatched["duration"].mean(),
                "mean_vp_cos_sim": df_unmatched["vp_cos_sim"].mean(),
                "earliest_start": df_unmatched["start_time"].min(),
                "latest_end": df_unmatched["end_time"].max(),
            })

        df_match_summary = pl.DataFrame(match_summary_rows)
        match_detail_path = os.path.join(output_dir, f"vad_vp_match_detail_{audio_id}.csv")
        match_summary_path = os.path.join(output_dir, f"vad_vp_match_summary_{audio_id}.csv")
        df.sort(["vp_matched", "vp_speaker", "start_time"], descending=[True, False, False]).write_csv(match_detail_path)
        df_match_summary.write_csv(match_summary_path)
        print(f"  VP照合 詳細 CSV: {match_detail_path}")
        print(f"  VP照合 サマリ CSV: {match_summary_path}")
        print(df_match_summary)

    return df


def plot_loudness_histogram(df: pl.DataFrame, save_path: str = "loudness_hist.png") -> None:
    """train / test を同じ bin で色分けした loudness のヒストグラムを描画・保存する。"""
    import numpy as np

    all_values = df["loudness"].to_numpy()
    bins = np.linspace(all_values.min(), all_values.max(), 31)

    splits = df["split"].unique().sort().to_list()

    fig, ax = plt.subplots(figsize=(10, 5))
    for split in splits:
        sub = df.filter(pl.col("split") == split)
        values = sub["loudness"].to_list()
        mean_val = sub["loudness"].mean()
        ax.hist(values, bins=bins, edgecolor="black", alpha=0.5, label=f"{split} (mean={mean_val:.1f})")
        ax.axvline(mean_val, linestyle="--", linewidth=1)

    ax.set_xlabel("Loudness (LUFS)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Loudness (train / test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ヒストグラムを {save_path} に保存しました")


def plot_loudness_bar(df: pl.DataFrame, save_path: str = "loudness_bar.png") -> None:
    """train / test 別に、各ファイルの loudness を棒グラフで描画・保存する。"""
    splits = df["split"].unique().sort().to_list()
    n_splits = len(splits)

    fig, axes = plt.subplots(n_splits, 1, figsize=(14, 5 * n_splits), squeeze=False)
    for ax, split in zip(axes[:, 0], splits):
        sub = df.filter(pl.col("split") == split).sort("audio_id")
        ids = sub["audio_id"].to_list()
        values = sub["loudness"].to_list()

        ax.bar(range(len(ids)), values, edgecolor="black", alpha=0.7)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90, fontsize=6)
        ax.set_ylabel("Loudness (LUFS)")
        ax.set_title(f"Loudness per File ({split})")
        ax.axhline(sub["loudness"].mean(), color="red", linestyle="--", linewidth=1,
                    label=f"mean={sub['loudness'].mean():.1f}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"棒グラフを {save_path} に保存しました")


def _plot_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    title: str,
    save_path: str,
    n_mels: int = SPECTROGRAM_N_MELS,
    fmax: float = SPECTROGRAM_FMAX,
    dpi: int = SPECTROGRAM_DPI,
) -> None:
    """波形 y からメルスペクトログラムを描画して保存する（共通処理）。"""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db, ax=ax, sr=sr, x_axis="time", y_axis="mel", fmax=fmax
    )
    ax.set_ylim(0, fmax)
    ax.set_yticks([0, 250, 500, 750, 1000, 1500, 2000, 4000, 6000, 8000])
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _lpc_envelope_and_formants(
    chunk: np.ndarray,
    sr: int,
    order: int = LPC_ORDER,
    n_fft: int = 8192,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], bool]:
    """LPC 分析でスペクトル包絡線とフォルマント周波数を返す。

    Returns:
        (freqs, envelope_db, formant_freqs, formant_bws, is_voiced)
        is_voiced が False の場合、フォルマントは信頼できない
    """
    rms_db = 20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-12)
    is_voiced = rms_db > LPC_MIN_ENERGY_DB

    # プリエンファシス
    if LPC_PREEMPHASIS > 0:
        chunk = np.append(chunk[0], chunk[1:] - LPC_PREEMPHASIS * chunk[:-1])

    windowed = chunk * np.hamming(len(chunk))
    a = librosa.lpc(windowed, order=order)

    # LPC スペクトル包絡
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    z_arr = np.exp(-1j * 2 * np.pi * freqs / sr)
    h = np.array([1.0 / np.polyval(a, z) for z in z_arr])
    envelope_db = 20 * np.log10(np.abs(h) + 1e-12)

    if not is_voiced:
        return freqs, envelope_db, [], [], False

    # 根からフォルマント周波数・帯域幅を推定
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]

    formant_freqs: list[float] = []
    formant_bws: list[float] = []
    for root in roots:
        freq = np.arctan2(np.imag(root), np.real(root)) * (sr / (2 * np.pi))
        bw = -sr / (2 * np.pi) * np.log(np.abs(root) + 1e-12)
        if (LPC_FORMANT_MIN_FREQ < freq < LPC_FORMANT_MAX_FREQ
                and LPC_BW_MIN < bw < LPC_BW_MAX):
            formant_freqs.append(freq)
            formant_bws.append(bw)

    sorted_idx = np.argsort(formant_freqs)
    formant_freqs = [formant_freqs[i] for i in sorted_idx][:LPC_MAX_FORMANTS]
    formant_bws = [formant_bws[i] for i in sorted_idx][:LPC_MAX_FORMANTS]

    return freqs, envelope_db, formant_freqs, formant_bws, is_voiced


def _plot_decay_fit_residual(
    freqs: np.ndarray,
    power: np.ndarray,
    stem: str,
    center_sec: float,
    window_sec: float,
    freq_min: float,
    freq_max: float,
    out_dir: str,
) -> None:
    """power（smoothed パワースペクトル）を A*exp(-B*f)+C でフィットし、残差を別画像で出力。"""
    from scipy.optimize import curve_fit

    fit_max = min(freq_max, DECAY_FIT_FREQ_MAX)
    mask = freqs <= fit_max
    freqs_fit = freqs[mask]
    power_fit = power[mask]
    if len(freqs_fit) < 10:
        print("  ⚠ フィット区間のサンプル数が少なすぎます")
        return

    def decay_model(f, a, b, c):
        return a * np.exp(-b * f) + c

    try:
        p0 = [power_fit[0] - power_fit[-1], 1e-3, float(np.median(power_fit))]
        popt, _ = curve_fit(decay_model, freqs_fit, power_fit, p0=p0, maxfev=10000)
        fitted = decay_model(freqs_fit, *popt)
    except RuntimeError:
        print("  ⚠ 減衰モデルのフィッティングに失敗しました")
        return

    residual = power_fit - fitted

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})

    ax_top = axes[0]
    ax_top.plot(freqs_fit, power_fit, linewidth=1.0, color="tab:blue", label="smoothed")
    ax_top.plot(freqs_fit, fitted, linewidth=1.5, color="tab:orange", linestyle="--",
                label=f"fit: {popt[0]:.1f}·exp(−{popt[1]:.4f}·f){popt[2]:+.1f}")
    ax_top.set_ylabel("Power (dB)")
    ax_top.set_title(f"Decay fit — {stem}  t={center_sec:.3f}s (window={window_sec*1000:.0f}ms, f≤{fit_max:.0f}Hz)")
    ax_top.set_xlim(freq_min, fit_max)
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    ax_bot = axes[1]
    ax_bot.plot(freqs_fit, residual, linewidth=0.8, color="tab:red")
    ax_bot.axhline(0, color="black", linewidth=0.5)
    ax_bot.set_xlabel("Frequency (Hz)")
    ax_bot.set_ylabel("Residual (dB)")
    ax_bot.set_xlim(freq_min, fit_max)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_title(f"Residual (RMSE={np.sqrt(np.mean(residual**2)):.2f} dB)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{center_sec:.3f}s_residual.png")
    fig.savefig(out_path, dpi=SPECTROGRAM_DPI)
    plt.close(fig)
    print(f"  フィット: A={popt[0]:.1f}, B={popt[1]:.4f}, C={popt[2]:.1f}  RMSE={np.sqrt(np.mean(residual**2)):.2f} dB")
    print(f"  残差プロット保存: {out_path}")


def plot_power_spectrum(
    wav_path: str,
    center_sec: float = SPECTRUM_CENTER_SEC,
    window_sec: float = SPECTRUM_WINDOW_SEC,
    output_dir: str | None = None,
    n_fft: int | None = SPECTRUM_N_FFT,
    freq_min: float = SPECTRUM_FREQ_MIN,
    freq_max: float = SPECTRUM_FREQ_MAX,
) -> str:
    """指定時刻のパワースペクトルを出力する（横軸: 周波数、縦軸: パワー dB）。

    Returns:
        保存した画像のパス
    """
    wav_path = os.path.normpath(wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {wav_path}")

    y, sr = librosa.load(wav_path, sr=SPECTROGRAM_SR, mono=True)
    half_win = window_sec / 2.0
    start_sample = max(0, int((center_sec - half_win) * sr))
    end_sample = min(len(y), int((center_sec + half_win) * sr))
    chunk = y[start_sample:end_sample].astype(np.float64)

    if n_fft is None:
        n_fft = len(chunk)
    n_fft = max(n_fft, len(chunk))

    spectrum = np.abs(np.fft.rfft(chunk, n=n_fft))
    power_db = 20 * np.log10(spectrum + 1e-12)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_plot = freqs[mask]
    power_raw = power_db[mask]

    power_smooth = power_raw
    if SPECTRUM_BANDPASS_ENABLED:
        P = np.fft.rfft(power_smooth)
        low = max(1, int(len(P) * SPECTRUM_BANDPASS_KEEP_LOW))
        high = max(low + 1, int(len(P) * SPECTRUM_BANDPASS_KEEP_HIGH))
        P[:low] = 0
        P[high:] = 0
        power_smooth = np.fft.irfft(P, n=len(power_smooth)).real
    if SPECTRUM_MA_ENABLED and SPECTRUM_MA_WINDOW > 1:
        kernel = np.ones(SPECTRUM_MA_WINDOW) / SPECTRUM_MA_WINDOW
        power_smooth = np.convolve(power_smooth, kernel, mode="same")
    if SPECTRUM_SMOOTH_ENABLED and SPECTRUM_SMOOTH_WINDOW > 0 and len(power_smooth) > SPECTRUM_SMOOTH_WINDOW:
        from scipy.signal import savgol_filter
        power_smooth = savgol_filter(power_smooth, SPECTRUM_SMOOTH_WINDOW, SPECTRUM_SMOOTH_POLYORDER)

    stem = Path(wav_path).stem
    base_dir = output_dir or SPECTROGRAM_OUTPUT_DIR or os.path.join(os.path.dirname(wav_path), "spectrograms")
    out_dir = os.path.join(base_dir, "partial_spectrum", stem)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{center_sec:.3f}s.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs_plot, power_raw, linewidth=0.5, alpha=0.35, color="gray", label="raw")
    ax.plot(freqs_plot, power_smooth, linewidth=1.2, color="tab:blue", label="smoothed")

    if LPC_ENABLED:
        lpc_freqs, lpc_env, formants, formant_bws, is_voiced = _lpc_envelope_and_formants(
            chunk, sr, order=LPC_ORDER, n_fft=n_fft
        )
        lpc_mask = (lpc_freqs >= freq_min) & (lpc_freqs <= freq_max)
        gain = np.median(power_raw[:len(power_raw)//4]) - np.median(lpc_env[lpc_mask][:len(power_raw)//4])
        ax.plot(lpc_freqs[lpc_mask], lpc_env[lpc_mask] + gain,
                linewidth=1.5, color="tab:red", alpha=0.8, label="LPC envelope")

        if not is_voiced:
            print("  ⚠ 無声/弱声フレームのためフォルマント検出スキップ")
        elif not formants:
            print("  フォルマント: 検出なし（帯域幅条件を満たす極なし）")
        else:
            for i, (f, bw) in enumerate(zip(formants, formant_bws)):
                ax.axvline(f, color="tab:green", linestyle="--", linewidth=1, alpha=0.7)
                ax.annotate(f"F{i+1}\n{f:.0f}Hz\n(bw={bw:.0f})", xy=(f, ax.get_ylim()[1]),
                            xytext=(5, -15), textcoords="offset points",
                            fontsize=8, color="tab:green", fontweight="bold")
            print(f"  フォルマント: {', '.join(f'F{i+1}={f:.0f}Hz (bw={bw:.0f})' for i, (f, bw) in enumerate(zip(formants, formant_bws)))}")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"Power spectrum — {stem}  t={center_sec:.3f}s (window={window_sec*1000:.0f}ms)")
    ax.set_xlim(freq_min, freq_max)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SPECTROGRAM_DPI)
    plt.close(fig)
    print(f"保存: {output_path}")

    if DECAY_FIT_ENABLED:
        residual_dir = os.path.join(base_dir, "spectrum_residual", stem)
        os.makedirs(residual_dir, exist_ok=True)
        _plot_decay_fit_residual(
            freqs_plot, power_smooth, stem, center_sec, window_sec,
            freq_min, freq_max, residual_dir,
        )

    return output_path


def plot_power_spectrum_scan(
    wav_path: str,
    step_sec: float = SPECTRUM_STEP_SEC,
    window_sec: float = SPECTRUM_WINDOW_SEC,
    output_dir: str | None = None,
) -> None:
    """音声クリップ全区間に対して step_sec ずつスライドしながらパワースペクトルを出力する。"""
    wav_path = os.path.normpath(wav_path)
    y, sr = librosa.load(wav_path, sr=SPECTROGRAM_SR, mono=True)
    duration = len(y) / sr
    half_win = window_sec / 2.0
    center = half_win
    count = 0
    while center + half_win <= duration:
        plot_power_spectrum(wav_path, center_sec=center, window_sec=window_sec, output_dir=output_dir)
        center += step_sec
        count += 1
    print(f"全区間スキャン完了: {count} フレーム出力")


def plot_voiceprint_spectrograms(
    voiceprints_dir: str,
    output_dir: str | None = None,
) -> None:
    """指定した voiceprints フォルダ内の各話者 wav のスペクトログラムを画像で出力する。
    さらに、同じ audio_id の元音声ファイル（親ディレクトリの {audio_id}.wav）の
    メルスペクトログラムも同じ mel フォルダに original.png として出力する。"""
    voiceprints_dir = os.path.normpath(voiceprints_dir)
    base_dir = output_dir or SPECTROGRAM_OUTPUT_DIR or os.path.join(os.path.dirname(voiceprints_dir), "spectrograms")
    mel_dir = os.path.join(os.path.normpath(base_dir), "mel")
    vad_mel_dir = os.path.join(mel_dir, "vad")
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(vad_mel_dir, exist_ok=True)
    # 出力先フォルダの既存画像を削除してから出力
    for p in glob.glob(os.path.join(mel_dir, "*.png")):
        try:
            os.remove(p)
        except OSError:
            pass
    for p in glob.glob(os.path.join(vad_mel_dir, "*.png")):
        try:
            os.remove(p)
        except OSError:
            pass

    wav_paths = sorted(glob.glob(os.path.join(voiceprints_dir, "*.wav")))
    if not wav_paths:
        print(f"wav が見つかりません: {voiceprints_dir}")
        return

    for wav_path in wav_paths:
        speaker_id = Path(wav_path).stem
        y, sr = librosa.load(wav_path, sr=SPECTROGRAM_SR, mono=True)
        out_path = os.path.join(mel_dir, f"{speaker_id}.png")
        _plot_mel_spectrogram(
            y, sr, f"Mel spectrogram — {speaker_id}", out_path
        )
        print(f"保存: {out_path}")

    # 元音声データのメルスペクトログラムを 2 分単位で同じ mel に出力
    n_orig = 0
    n_vad = 0
    segment_sec = 60  # 1 分
    parent_dir = Path(voiceprints_dir).resolve().parent
    audio_id = parent_dir.name
    original_wav = parent_dir / f"{audio_id}.wav"
    if original_wav.is_file():
        y, sr = librosa.load(str(original_wav), sr=SPECTROGRAM_SR, mono=True)
        duration_sec = len(y) / sr
        seg_samples = segment_sec * sr
        seg_idx = 0
        start_sample = 0
        while start_sample < len(y):
            end_sample = min(start_sample + seg_samples, len(y))
            y_seg = y[start_sample:end_sample]
            start_sec = start_sample / sr
            end_sec = end_sample / sr
            out_path_orig = os.path.join(mel_dir, f"original_{seg_idx:04d}_{int(start_sec)}s.png")
            _plot_mel_spectrogram(
                y_seg, sr,
                f"Mel — {audio_id} (original {int(start_sec)}s–{int(end_sec)}s)",
                out_path_orig,
            )
            print(f"保存: {out_path_orig}")
            n_orig += 1
            start_sample = end_sample
            seg_idx += 1

        # VAD で切り取った区間のメルスペクトログラムを mel/vad/ に出力
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
        vad_model = load_silero_vad()
        audio_tensor = read_audio(str(original_wav))
        vad_segments = get_speech_timestamps(audio_tensor, vad_model, return_seconds=True)
        if vad_segments:
            filtered = []
            for seg in vad_segments:
                dur = seg["end"] - seg["start"]
                if VAD_MIN_DURATION_FILTER and dur < VAD_MIN_DURATION_SEC:
                    continue
                filtered.append(seg)
            vad_segments = filtered
        n_vad = 0
        for i, seg in enumerate(vad_segments or []):
            start_sec, end_sec = seg["start"], seg["end"]
            start_samp = int(start_sec * sr)
            end_samp = int(end_sec * sr)
            if start_samp >= len(y) or start_samp >= end_samp:
                continue
            end_samp = min(end_samp, len(y))
            y_vad = y[start_samp:end_samp]
            out_vad = os.path.join(vad_mel_dir, f"seg_{i:04d}_{start_sec:.1f}s_{end_sec:.1f}s.png")
            _plot_mel_spectrogram(
                y_vad, sr,
                f"Mel — {audio_id} VAD {start_sec:.1f}s–{end_sec:.1f}s",
                out_vad,
            )
            print(f"保存: {out_vad}")
            n_vad += 1
        if n_vad > 0:
            print(f"VAD 区間メルスペクトログラムを {vad_mel_dir} に {n_vad} 件出力しました")
    else:
        print(f"元音声が見つかりません（スキップ）: {original_wav}")

    print(f"スペクトログラムを {mel_dir} に {len(wav_paths)} 件 + 元音声 {n_orig} 件 + VAD {n_vad} 件出力しました")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if EDA_LOUDNESS:
        dfs: list[pl.DataFrame] = []
        for split in ("train", "test"):
            files = get_audio_files(split)
            print(f"{split} ファイル数: {len(files)}")
            if files:
                dfs.append(compute_loudness(files, split))

        df_loudness = pl.concat(dfs)
        print(df_loudness.describe())

        plot_loudness_histogram(df_loudness, save_path=os.path.join(OUTPUT_DIR, "loudness_hist.png"))
        plot_loudness_bar(df_loudness, save_path=os.path.join(OUTPUT_DIR, "loudness_bar.png"))

    if EDA_SEGMENT_LOUDNESS:
        dfs_seg: list[pl.DataFrame] = []
        for split in ("train", "test"):
            files = get_audio_files(split)
            if files:
                dfs_seg.append(compute_segment_loudness(files, split))
        if dfs_seg:
            df_seg = pl.concat(dfs_seg).sort(["split", "audio_id", "start_time"])
            seg_csv_path = os.path.join(OUTPUT_DIR, "segment_loudness.csv")
            df_seg.write_csv(seg_csv_path)
            print(f"segment loudness を {seg_csv_path} に保存しました ({len(df_seg)} 件)")

    if EDA_VOICEPRINT_LOUDNESS:
        dfs_vp: list[pl.DataFrame] = []
        for split in ("train", "test"):
            vp_files = glob.glob(os.path.join(INPUT_DIR, split, "*", "voiceprints", "[A-Z].wav"))
            if vp_files:
                dfs_vp.append(compute_voiceprint_loudness(split))
        if dfs_vp:
            df_vp = pl.concat(dfs_vp).sort(["split", "audio_id", "speaker"])
            vp_csv_path = os.path.join(OUTPUT_DIR, "voiceprint_loudness.csv")
            df_vp.write_csv(vp_csv_path)
            print(f"voiceprint loudness を {vp_csv_path} に保存しました ({len(df_vp)} 件)")

    if EDA_VAD_CLUSTERING or EDA_VAD_VOICEPRINT_MATCH:
        compute_vad_clustering(EDA_VAD_CLUSTERING_INPUT, output_dir=OUTPUT_DIR)

    print("EDA 処理が終わりました。")


def _parse_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _expand_input_path(pattern: str, expect_dir: bool = False) -> list[str]:
    """glob パターンを展開して対象パスのリストを返す。"""
    if any(c in pattern for c in ("*", "?", "[")):
        paths = sorted(glob.glob(pattern, recursive=True))
        if expect_dir:
            paths = [p for p in paths if os.path.isdir(p)]
        return paths
    normalized = os.path.normpath(pattern)
    if os.path.exists(normalized):
        return [normalized]
    return []


def _derive_output_dir(input_path: str) -> str:
    """入力パスから出力ディレクトリを導出する。

    パス中に ``train`` / ``test`` が含まれていれば
    ``{base}/train/{audio_id}`` のようにサブフォルダを切る。
    """
    if SPECTROGRAM_OUTPUT_DIR:
        base = os.path.dirname(os.path.normpath(SPECTROGRAM_OUTPUT_DIR))
    else:
        base = os.path.join(".", "eda", "spectrograms")
    p = Path(input_path).resolve()
    if p.name == "voiceprints":
        audio_id = p.parent.name
    elif p.suffix:
        audio_id = p.parent.parent.name if p.parent.name == "voiceprints" else p.parent.name
    else:
        audio_id = p.name
    parts = p.parts
    split = None
    for part in parts:
        if part in ("train", "test"):
            split = part
            break
    if split:
        return os.path.join(base, split, audio_id)
    return os.path.join(base, audio_id)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        # CLI 引数があればそちらを優先（従来互換）
        path = os.path.normpath(sys.argv[1])
        args = sys.argv[2:]
        if os.path.isdir(path):
            if SPECTROGRAM_FOLDER:
                output_dir = args[0] if args else None
                plot_voiceprint_spectrograms(path, output_dir=output_dir)
        elif os.path.isfile(path):
            if SPECTROGRAM_SINGLE:
                if SPECTRUM_SCAN_ALL and len(args) == 0:
                    plot_power_spectrum_scan(path)
                else:
                    center = _parse_float(args[0]) if len(args) >= 1 and _parse_float(args[0]) is not None else SPECTRUM_CENTER_SEC
                    window = _parse_float(args[1]) if len(args) >= 2 and _parse_float(args[1]) is not None else SPECTRUM_WINDOW_SEC
                    plot_power_spectrum(path, center_sec=center, window_sec=window)
        else:
            print(f"パスが見つかりません: {path}")
        print("EDA 処理が終わりました。")
    else:
        ran_any = False

        if MEL_INPUT_PATH:
            mel_dirs = _expand_input_path(MEL_INPUT_PATH, expect_dir=True)
            if mel_dirs:
                for d in mel_dirs:
                    out = _derive_output_dir(d)
                    print(f"メルスペクトログラム対象: {d}  →  {out}")
                    plot_voiceprint_spectrograms(d, output_dir=out)
                ran_any = True

        if SPECTROGRAM_SINGLE and SPECTRUM_INPUT_PATH:
            spec_paths = _expand_input_path(SPECTRUM_INPUT_PATH)
            for sp in spec_paths:
                if os.path.isfile(sp):
                    out = _derive_output_dir(sp)
                    if SPECTRUM_SCAN_ALL:
                        plot_power_spectrum_scan(sp, output_dir=out)
                    else:
                        plot_power_spectrum(sp, output_dir=out)
                    ran_any = True
                elif os.path.isdir(sp):
                    out = _derive_output_dir(sp)
                    wav_files = sorted(glob.glob(os.path.join(sp, "*.wav")))
                    print(f"パワースペクトル対象: {len(wav_files)} ファイル ({sp})")
                    for wf in wav_files:
                        if SPECTRUM_SCAN_ALL:
                            plot_power_spectrum_scan(wf, output_dir=out)
                        else:
                            plot_power_spectrum(wf, output_dir=out)
                    ran_any = True

        if not ran_any:
            main()
        else:
            print("EDA 処理が終わりました。")
