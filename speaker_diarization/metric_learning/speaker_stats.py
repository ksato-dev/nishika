"""クラスごとの話者時間統計を表示するスクリプト。

使い方:
  python metric_learning/speaker_stats.py
  python metric_learning/speaker_stats.py --annotation input/train_annotation.csv --top 30
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def parse_args():
    p = argparse.ArgumentParser(description="クラス (audio_id__speaker) ごとの発話時間統計")
    p.add_argument("--annotation", type=str, default="input/train_annotation.csv")
    p.add_argument("--top", type=int, default=20, help="上位/下位に表示するクラス数")
    p.add_argument("--sort", choices=["total", "segments", "mean_seg"], default="total",
                   help="ソート基準")
    return p.parse_args()


def main():
    args = parse_args()

    df = pl.read_csv(args.annotation)
    df = df.with_columns(
        (pl.col("end_time") - pl.col("start_time")).alias("duration"),
        (pl.col("audio_id") + "__" + pl.col("speaker")).alias("class_name"),
    )

    sort_col = {
        "total": "total_sec",
        "segments": "n_segments",
        "mean_seg": "mean_seg_sec",
    }[args.sort]

    cls_stats = df.group_by("class_name").agg(
        pl.col("duration").sum().alias("total_sec"),
        pl.col("duration").count().alias("n_segments"),
        pl.col("duration").mean().alias("mean_seg_sec"),
        pl.col("duration").median().alias("median_seg_sec"),
        pl.col("duration").min().alias("min_seg_sec"),
        pl.col("duration").max().alias("max_seg_sec"),
    ).sort(sort_col, descending=True)

    print("=== クラスごとの話者時間 統計 ===")
    print(f"クラス数: {len(cls_stats)}")

    totals = cls_stats["total_sec"].to_numpy()
    print()
    print("--- クラスあたり合計発話時間 (秒) ---")
    print(f"  mean  = {totals.mean():.2f}")
    print(f"  median= {np.median(totals):.2f}")
    print(f"  std   = {totals.std():.2f}")
    print(f"  min   = {totals.min():.2f}")
    print(f"  max   = {totals.max():.2f}")
    print(f"  <5s   : {(totals < 5).sum()} classes")
    print(f"  <10s  : {(totals < 10).sum()} classes")
    print(f"  <30s  : {(totals < 30).sum()} classes")
    print(f"  >60s  : {(totals > 60).sum()} classes")
    print(f"  >120s : {(totals > 120).sum()} classes")
    print(f"  >300s : {(totals > 300).sum()} classes")

    segs = cls_stats["mean_seg_sec"].to_numpy()
    print()
    print("--- セグメントあたり平均時間 (秒) ---")
    print(f"  mean  = {segs.mean():.2f}")
    print(f"  median= {np.median(segs):.2f}")
    print(f"  std   = {segs.std():.2f}")
    print(f"  min   = {segs.min():.2f}")
    print(f"  max   = {segs.max():.2f}")

    n_segs = cls_stats["n_segments"].to_numpy()
    print()
    print("--- クラスあたりセグメント数 ---")
    print(f"  mean  = {n_segs.mean():.1f}")
    print(f"  median= {np.median(n_segs):.0f}")
    print(f"  min   = {n_segs.min()}")
    print(f"  max   = {n_segs.max()}")

    n = args.top
    print()
    print(f"--- 上位{n}クラス ({sort_col} 降順) ---")
    for row in cls_stats.head(n).iter_rows(named=True):
        cn = row["class_name"]
        ts = row["total_sec"]
        ns = row["n_segments"]
        ms = row["mean_seg_sec"]
        print(f"  {cn:30s}  total={ts:7.1f}s  segs={ns:4d}  mean_seg={ms:.2f}s")

    print()
    print(f"--- 下位{n}クラス ({sort_col} 昇順) ---")
    for row in cls_stats.tail(n).iter_rows(named=True):
        cn = row["class_name"]
        ts = row["total_sec"]
        ns = row["n_segments"]
        ms = row["mean_seg_sec"]
        print(f"  {cn:30s}  total={ts:7.1f}s  segs={ns:4d}  mean_seg={ms:.2f}s")

    print()
    print("--- 合計発話時間の分布 (ヒストグラム) ---")
    bins = [0, 2, 5, 10, 20, 30, 60, 120, 300, 600, float("inf")]
    labels = ["0-2", "2-5", "5-10", "10-20", "20-30", "30-60", "60-120", "120-300", "300-600", "600+"]
    for i in range(len(bins) - 1):
        cnt = int(((totals >= bins[i]) & (totals < bins[i + 1])).sum())
        bar = "#" * (cnt // 2)
        print(f"  {labels[i]:>8s}s : {cnt:4d} {bar}")


if __name__ == "__main__":
    main()
