"""アノテーション CSV + voiceprint から話者ごとの音声クリップを抽出する。

出力構成:
  metric_learning/data/<audio_id>__<speaker>/
    seg_0001.wav
    seg_0002.wav
    ...
    voiceprint.wav   (存在する場合)

クラス = (audio_id, speaker) の組。異なるレコードの同一話者 ID は別クラス。
"""

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def load_annotations(csv_path: str) -> dict[str, list[dict]]:
    """audio_id ごとにセグメント一覧を返す。"""
    records: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row["audio_id"]].append({
                "start": float(row["start_time"]),
                "end": float(row["end_time"]),
                "speaker": row["speaker"],
            })
    return dict(records)


def extract(
    annotation_csv: str,
    train_dir: str,
    output_dir: str,
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    target_sr: int = 16_000,
    include_voiceprints: bool = True,
):
    annotations = load_annotations(annotation_csv)
    os.makedirs(output_dir, exist_ok=True)

    total_clips = 0
    total_skipped = 0
    class_counts: dict[str, int] = defaultdict(int)

    for audio_id, segments in sorted(annotations.items()):
        wav_path = os.path.join(train_dir, audio_id, f"{audio_id}.wav")
        if not os.path.isfile(wav_path):
            print(f"  [SKIP] wav not found: {wav_path}")
            continue

        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_len = len(audio) / sr

        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration < min_duration:
                total_skipped += 1
                continue

            class_name = f"{audio_id}__{seg['speaker']}"
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            start_sample = int(seg["start"] * sr)
            end_sample = int(min(seg["end"], audio_len) * sr)
            clip = audio[start_sample:end_sample]

            if len(clip) < int(min_duration * sr):
                total_skipped += 1
                continue

            # 長すぎるセグメントは max_duration でトリム
            max_samples = int(max_duration * sr)
            if len(clip) > max_samples:
                clip = clip[:max_samples]

            class_counts[class_name] += 1
            idx = class_counts[class_name]
            out_path = os.path.join(class_dir, f"seg_{idx:04d}.wav")
            sf.write(out_path, clip, sr)
            total_clips += 1

        # voiceprints
        if include_voiceprints:
            vp_dir = os.path.join(train_dir, audio_id, "voiceprints")
            if os.path.isdir(vp_dir):
                for vp_file in sorted(os.listdir(vp_dir)):
                    if not vp_file.endswith(".wav"):
                        continue
                    speaker = Path(vp_file).stem
                    class_name = f"{audio_id}__{speaker}"
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    src = os.path.join(vp_dir, vp_file)
                    dst = os.path.join(class_dir, "voiceprint.wav")
                    shutil.copy2(src, dst)
                    total_clips += 1

        print(f"  {audio_id}: {len(segments)} segments")

    n_classes = len(set(class_counts.keys()))
    # voiceprint-only クラスも数える
    all_class_dirs = [d for d in os.listdir(output_dir)
                      if os.path.isdir(os.path.join(output_dir, d))]
    print(f"\nDone: {total_clips} clips, {len(all_class_dirs)} classes, "
          f"{total_skipped} segments skipped (< {min_duration}s)")

    # クラスごとのサンプル数統計
    counts = []
    for d in all_class_dirs:
        n = len([f for f in os.listdir(os.path.join(output_dir, d)) if f.endswith(".wav")])
        counts.append(n)
    counts_arr = np.array(counts)
    print(f"Samples per class: min={counts_arr.min()}, max={counts_arr.max()}, "
          f"mean={counts_arr.mean():.1f}, median={np.median(counts_arr):.0f}")


def main():
    p = argparse.ArgumentParser(description="Extract speaker clips from annotations")
    p.add_argument("--annotation_csv",  default="./input/train_annotation.csv")
    p.add_argument("--train_dir",       default="./input/train")
    p.add_argument("--output_dir",      default="./metric_learning/data")
    p.add_argument("--min_duration",    type=float, default=0.5,
                   help="Minimum segment duration in seconds (default: 0.5)")
    p.add_argument("--max_duration",    type=float, default=15.0,
                   help="Maximum segment duration in seconds (default: 15.0)")
    p.add_argument("--no_voiceprints",  action="store_true",
                   help="Do not include voiceprint wav files")
    args = p.parse_args()

    extract(
        annotation_csv=args.annotation_csv,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        include_voiceprints=not args.no_voiceprints,
    )


if __name__ == "__main__":
    main()
