"""K-fold 交差検証 + 平均ベストエポックで全データ学習。

使い方:
  python metric_learning/cross_validate.py \
    --data_dir ./metric_learning/data \
    --output_dir ./metric_learning/checkpoints \
    --epochs 50 --batch_size 24 --lr 3e-4 \
    --warmup_epochs 3 \
    --margin_schedule "0,0.05,0.1,0.15,0.2,0.25,0.3,0.35" \
    --scale_schedule "20,32,40,48" \
    --schedule_epochs 24 \
    --max_samples_per_class 100 --balance_classes --balance_samples_per_class 20 \
    --gradient_checkpointing \
    --init_weights ./metric_learning/checkpoints/checkpoint_epoch005.pt \
    --n_folds 5 --early_stopping_patience 5 \
    --log_file ./metric_learning/checkpoints/cv_log.txt
"""

import copy
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import parse_args, setup_logging, train

logger = logging.getLogger(__name__)


def main():
    base_args = parse_args()
    setup_logging(base_args.log_file)

    n_folds = base_args.n_folds
    logger.info("=" * 80)
    logger.info("Cross-Validation: %d folds, early_stopping_patience=%d",
                n_folds, base_args.early_stopping_patience)
    logger.info("=" * 80)

    fold_results = []
    cv_base_dir = os.path.join(base_args.output_dir, "cv")
    os.makedirs(cv_base_dir, exist_ok=True)

    for fold_idx in range(n_folds):
        logger.info("")
        logger.info("=" * 80)
        logger.info("FOLD %d / %d", fold_idx + 1, n_folds)
        logger.info("=" * 80)

        fold_args = copy.deepcopy(base_args)
        fold_args.fold = fold_idx
        fold_args.n_folds = n_folds
        fold_args.train_all = False
        fold_args.output_dir = os.path.join(cv_base_dir, f"fold_{fold_idx}")
        fold_args.log_file = base_args.log_file
        os.makedirs(fold_args.output_dir, exist_ok=True)

        start_time = time.time()
        result = train(fold_args)
        elapsed = time.time() - start_time

        fold_results.append({
            "fold": fold_idx,
            "best_epoch": result["best_epoch"],
            "best_val_acc": result["best_val_acc"],
            "elapsed_sec": elapsed,
        })
        logger.info("Fold %d: best_epoch=%d, best_val_acc=%.4f (%.1f min)",
                     fold_idx, result["best_epoch"], result["best_val_acc"],
                     elapsed / 60)

        # GPU メモリ解放
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- CV サマリー ---
    best_epochs = [r["best_epoch"] for r in fold_results]
    best_accs = [r["best_val_acc"] for r in fold_results]
    avg_best_epoch = int(round(np.mean(best_epochs)))
    avg_acc = np.mean(best_accs)

    logger.info("")
    logger.info("=" * 80)
    logger.info("CV Summary:")
    for r in fold_results:
        logger.info("  Fold %d: best_epoch=%d, val_acc=%.4f",
                     r["fold"], r["best_epoch"], r["best_val_acc"])
    logger.info("  Avg best epoch: %d (raw: %s)", avg_best_epoch, best_epochs)
    logger.info("  Avg val acc: %.4f (std=%.4f)", avg_acc, np.std(best_accs))
    logger.info("=" * 80)

    cv_summary = {
        "fold_results": fold_results,
        "avg_best_epoch": avg_best_epoch,
        "avg_val_acc": avg_acc,
    }
    with open(os.path.join(cv_base_dir, "cv_summary.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)

    # --- 全データで avg_best_epoch まで学習 ---
    logger.info("")
    logger.info("=" * 80)
    logger.info("Final Training: all data, %d epochs (avg best from CV)", avg_best_epoch)
    logger.info("=" * 80)

    final_args = copy.deepcopy(base_args)
    final_args.fold = None
    final_args.train_all = True
    final_args.epochs = avg_best_epoch
    final_args.early_stopping_patience = 0
    final_args.output_dir = os.path.join(base_args.output_dir, "final")
    final_args.log_file = base_args.log_file
    os.makedirs(final_args.output_dir, exist_ok=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_result = train(final_args)

    logger.info("")
    logger.info("=" * 80)
    logger.info("All done!")
    logger.info("  CV avg best epoch: %d", avg_best_epoch)
    logger.info("  CV avg val acc: %.4f", avg_acc)
    logger.info("  Final model: %s/final_model.pt", final_args.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
