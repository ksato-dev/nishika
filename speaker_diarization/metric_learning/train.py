"""wav2vec2 + ArcFace 話者メトリック学習 学習スクリプト。

使い方:
  python metric_learning/extract_clips.py   # まずクリップ抽出
  python metric_learning/train.py --data_dir ./metric_learning/data --output_dir ./metric_learning/checkpoints
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    SpeakerClipDataset,
    collate_fn,
    build_label_map,
    WaveformAugmentor,
    HardNegativeBatchSampler,
    ClassBalancedSampler,
)
from model import SpeakerMetricLearner

logger = logging.getLogger(__name__)


class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_file=None):
    handlers = []
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(console_handler)
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = FlushingFileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train wav2vec2 + ArcFace speaker metric learning")
    p.add_argument("--data_dir", type=str, default="./metric_learning/data")
    p.add_argument("--output_dir", type=str, default="./metric_learning/checkpoints")
    p.add_argument("--pretrained_model", type=str, default="microsoft/wavlm-base-plus")
    p.add_argument("--embedding_dim", type=int, default=192)
    p.add_argument("--num_subcenters", type=int, default=3,
                   help="SubCenter ArcFace のサブセンター数")
    p.add_argument("--arcface_scale", type=float, default=20.0)
    p.add_argument("--arcface_margin", type=float, default=0.0)
    p.add_argument("--freeze_feature_extractor", action="store_true", default=True)
    p.add_argument("--freeze_transformer_layers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--init_weights", type=str, default=None,
                   help="初期重みチェックポイント (転移学習用)")
    p.add_argument("--resume", type=str, default=None,
                   help="チェックポイントから学習を再開 (model + optimizer + scheduler + epoch)")

    # Augmentation
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--aug_noise_prob", type=float, default=0.5)
    p.add_argument("--aug_noise_snr_min", type=float, default=10)
    p.add_argument("--aug_noise_snr_max", type=float, default=40)
    p.add_argument("--aug_gain_prob", type=float, default=0.5)
    p.add_argument("--aug_gain_db_min", type=float, default=-6)
    p.add_argument("--aug_gain_db_max", type=float, default=6)
    p.add_argument("--aug_time_shift_prob", type=float, default=0.3)
    p.add_argument("--aug_time_shift_max_ratio", type=float, default=0.1)
    p.add_argument("--aug_speed_prob", type=float, default=0.3)
    p.add_argument("--aug_speed_min", type=float, default=0.9)
    p.add_argument("--aug_speed_max", type=float, default=1.1)
    p.add_argument("--aug_time_mask_prob", type=float, default=0.3)
    p.add_argument("--aug_time_mask_max_ratio", type=float, default=0.1)
    p.add_argument("--aug_time_mask_num", type=int, default=2)
    p.add_argument("--aug_polarity_prob", type=float, default=0.2)
    p.add_argument("--aug_pitch_prob", type=float, default=0.0)
    p.add_argument("--aug_pitch_range_semitones", type=float, default=2.0)
    p.add_argument("--aug_reverb_prob", type=float, default=0.0)
    p.add_argument("--aug_reverb_decay", type=float, default=4.0)
    p.add_argument("--aug_reverb_room_size", type=float, default=0.3)
    p.add_argument("--aug_reverb_wet", type=float, default=0.3)

    # Schedule
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--margin_warmup_epochs", type=int, default=0)
    p.add_argument("--margin_schedule", type=str, default=None,
                   help="カンマ区切りの margin スケジュール。例: 0,0.05,0.1,0.2,0.3")
    p.add_argument("--scale_schedule", type=str, default=None,
                   help="カンマ区切りの scale スケジュール。例: 20,32,40")
    p.add_argument("--schedule_epochs", type=int, default=10)

    # Quick run
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--val_interval", type=int, default=1)

    # Checkpointing
    p.add_argument("--save_interval", type=int, default=5,
                   help="N エポックごとに定期チェックポイントを保存 (0=off)")

    # Performance
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--grad_accum_steps", type=int, default=1)

    # Hard Negative Mining
    p.add_argument("--hard_negative", action="store_true")
    p.add_argument("--hn_p_classes", type=int, default=6)
    p.add_argument("--hn_k_samples", type=int, default=4)
    p.add_argument("--hn_hard_ratio", type=float, default=0.5)
    p.add_argument("--hn_update_interval", type=int, default=5)
    p.add_argument("--hn_start_epoch", type=int, default=None)
    p.add_argument("--hn_warmup_epochs", type=int, default=5)

    # Data balancing
    p.add_argument("--max_samples_per_class", type=int, default=None,
                   help="クラスあたりの最大サンプル数。超過分をランダムに間引く")
    p.add_argument("--balance_classes", action="store_true",
                   help="ClassBalancedSampler で各クラスから均等にサンプル")
    p.add_argument("--balance_samples_per_class", type=int, default=20,
                   help="balance_classes 時の 1 エポックあたりクラスあたりサンプル数")
    p.add_argument("--split_by_record", action="store_true", default=True,
                   help="レコード (audio_id) 単位で train/val を分割する (デフォルト: ON)")
    p.add_argument("--no_split_by_record", dest="split_by_record", action="store_false",
                   help="従来のサンプル単位 shuffle split に戻す")

    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_schedule(s):
    if s is None:
        return None
    vals = [x.strip() for x in s.split(",") if x.strip()]
    return [float(x) for x in vals] if vals else None


def _interpolate_schedule(values, epoch, schedule_epochs):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    schedule_epochs = max(int(schedule_epochs), 1)
    if epoch >= schedule_epochs:
        return values[-1]
    pos = (epoch - 1) / max(schedule_epochs - 1, 1) * (len(values) - 1)
    left = int(math.floor(pos))
    right = min(left + 1, len(values) - 1)
    alpha = pos - left
    return values[left] * (1.0 - alpha) + values[right] * alpha


def compute_class_similarity(model):
    """SubCenter ArcFace の重みから class 間コサイン類似度行列を計算。

    各クラスの K 個のサブセンターを平均して代表ベクトルにする。
    """
    with torch.no_grad():
        W = model.arcface.weight.data  # (C*K, D)
        K = model.arcface.num_subcenters
        C = model.arcface.num_classes
        W = W.view(C, K, -1).mean(dim=1)  # (C, D) — サブセンター平均
        W_norm = F.normalize(W, p=2, dim=1)
        return (W_norm @ W_norm.T).cpu().numpy()


def log_hard_negative_info(sim_matrix, class_names, top_k=5):
    n = sim_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim_matrix[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    lines = [f"  Top-{top_k} similar class pairs:"]
    for sim_val, a, b in pairs[:top_k]:
        lines.append(f"    cos={sim_val:.4f}  {a} <-> {b}")
    return "\n".join(lines)


def evaluate(model, val_loader, criterion, device, use_amp=False):
    """ArcFace ロジットベースの評価 (train と同じクラスセットの場合に使用)。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_values = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_values, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs["logits"], labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (total_loss / max(total, 1), correct / max(total, 1))


def evaluate_embeddings(model, val_loader, id2label, device, use_amp=False):
    """Embedding ベースの評価 (レコード単位 split で未知話者を評価)。

    最終用途に合わせ、レコード内の 3〜5 人の中から正しい話者を当てる精度
    (within-record nearest-centroid accuracy) を主指標とする。

    Returns:
        dict with keys: record_acc, intra_sim, inter_sim, sim_gap, n_records
    """
    from collections import defaultdict
    model.eval()
    raw_model = getattr(model, "_orig_mod", model)
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                emb = raw_model.extract_embedding(input_values, attention_mask=attention_mask)
            all_embeddings.append(emb.float().cpu())
            all_labels.append(batch["labels"])
            del input_values, attention_mask, emb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
    all_labels = torch.cat(all_labels, dim=0)           # (N,)

    # label_id → record_id のマッピング
    def _record_of(label_id):
        class_name = id2label.get(label_id, "")
        parts = class_name.split("__")
        return parts[0] if len(parts) >= 2 else class_name

    # レコード → {label_id: [sample_indices]} の構造に整理
    record_classes = defaultdict(lambda: defaultdict(list))
    for i, lab in enumerate(all_labels.tolist()):
        rec = _record_of(lab)
        record_classes[rec][lab].append(i)

    # --- レコード内 Nearest-Centroid Accuracy (leave-one-out) ---
    correct_total = 0
    sample_total = 0
    record_accs = []
    intra_sims_all = []
    inter_sims_all = []

    for rec_id, speakers in record_classes.items():
        valid_speakers = {s: idxs for s, idxs in speakers.items() if len(idxs) >= 2}
        if len(valid_speakers) < 2:
            continue

        speaker_labs = sorted(valid_speakers.keys())
        lab_to_local = {s: li for li, s in enumerate(speaker_labs)}

        # 各話者の centroid
        centroids = []
        for s in speaker_labs:
            idxs = valid_speakers[s]
            centroids.append(F.normalize(
                all_embeddings[idxs].mean(dim=0, keepdim=True), p=2, dim=1
            ))
        centroid_mat = torch.cat(centroids, dim=0)  # (S, D) where S = speakers in record

        rec_correct = 0
        rec_total = 0
        for s in speaker_labs:
            idxs = valid_speakers[s]
            local_idx = lab_to_local[s]
            embs_s = all_embeddings[idxs]  # (K, D)
            n_s = len(idxs)
            cls_sum = embs_s.sum(dim=0, keepdim=True)

            for k in range(n_s):
                emb_k = embs_s[k:k+1]  # (1, D)
                loo_cent = F.normalize((cls_sum - emb_k) / (n_s - 1), p=2, dim=1)
                ref = centroid_mat.clone()
                ref[local_idx] = loo_cent.squeeze(0)
                sims = F.cosine_similarity(emb_k, ref, dim=1)  # (S,)
                if sims.argmax().item() == local_idx:
                    rec_correct += 1
                rec_total += 1

        if rec_total > 0:
            record_accs.append(rec_correct / rec_total)
            correct_total += rec_correct
            sample_total += rec_total

        # Intra-class similarity (within this record)
        for s in speaker_labs:
            idxs = valid_speakers[s]
            if len(idxs) < 2:
                continue
            embs = all_embeddings[idxs]
            sims = F.cosine_similarity(embs.unsqueeze(0), embs.unsqueeze(1), dim=2)
            mask = ~torch.eye(len(idxs), dtype=torch.bool)
            intra_sims_all.append(sims[mask].mean().item())

        # Inter-class similarity (between speakers in this record)
        for i_a, s_a in enumerate(speaker_labs):
            for s_b in speaker_labs[i_a + 1:]:
                sim = F.cosine_similarity(centroids[lab_to_local[s_a]], centroids[lab_to_local[s_b]]).item()
                inter_sims_all.append(sim)

    record_acc = correct_total / max(sample_total, 1)
    intra_sim = float(np.mean(intra_sims_all)) if intra_sims_all else 0.0
    inter_sim = float(np.mean(inter_sims_all)) if inter_sims_all else 0.0

    return {
        "record_acc": record_acc,
        "record_acc_macro": float(np.mean(record_accs)) if record_accs else 0.0,
        "intra_sim": intra_sim,
        "inter_sim": inter_sim,
        "sim_gap": intra_sim - inter_sim,
        "n_records": len(record_accs),
    }


def train():
    args = parse_args()
    setup_logging(args.log_file)
    set_seed(args.seed)

    margin_schedule = _parse_schedule(args.margin_schedule)
    scale_schedule = _parse_schedule(args.scale_schedule)

    if args.hard_negative and args.hn_start_epoch is None:
        args.hn_start_epoch = args.schedule_epochs + 1

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    logger.info("AMP: %s", "ON" if use_amp else "OFF")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    label2id, class_names = build_label_map(args.data_dir)
    num_classes = len(label2id)
    logger.info("Classes: %d", num_classes)

    label_map_path = os.path.join(args.output_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "class_names": class_names}, f,
                  ensure_ascii=False, indent=2)

    augmentor = None
    if not args.no_augment:
        augmentor = WaveformAugmentor(
            noise_prob=args.aug_noise_prob,
            noise_snr_range=(args.aug_noise_snr_min, args.aug_noise_snr_max),
            gain_prob=args.aug_gain_prob,
            gain_db_range=(args.aug_gain_db_min, args.aug_gain_db_max),
            time_shift_prob=args.aug_time_shift_prob,
            time_shift_max_ratio=args.aug_time_shift_max_ratio,
            speed_prob=args.aug_speed_prob,
            speed_range=(args.aug_speed_min, args.aug_speed_max),
            time_mask_prob=args.aug_time_mask_prob,
            time_mask_max_ratio=args.aug_time_mask_max_ratio,
            time_mask_num=args.aug_time_mask_num,
            polarity_prob=args.aug_polarity_prob,
            pitch_prob=args.aug_pitch_prob,
            pitch_range_semitones=args.aug_pitch_range_semitones,
            reverb_prob=args.aug_reverb_prob,
            reverb_decay=args.aug_reverb_decay,
            reverb_room_size=args.aug_reverb_room_size,
            reverb_wet=args.aug_reverb_wet,
        )
        logger.info("Augmentation: ON")
    else:
        logger.info("Augmentation: OFF")

    train_dataset = SpeakerClipDataset(
        data_dir=args.data_dir, split="train",
        validation_ratio=args.val_ratio, label2id=label2id, seed=args.seed,
        augmentor=augmentor,
        max_samples_per_class=args.max_samples_per_class,
        split_by_record=args.split_by_record,
    )
    val_dataset = SpeakerClipDataset(
        data_dir=args.data_dir, split="val",
        validation_ratio=args.val_ratio, label2id=label2id, seed=args.seed,
        split_by_record=args.split_by_record,
    )

    if train_dataset._flat_audio is not None and args.num_workers > 0:
        args.num_workers = 0
        logger.info("Forced num_workers -> 0 (preload mode)")

    # --- サンプラー選択: HardNegative > ClassBalanced > shuffle ---
    batch_sampler = None
    class_balanced_sampler = None
    if args.hard_negative:
        batch_sampler = HardNegativeBatchSampler(
            labels=train_dataset.labels,
            p_classes=args.hn_p_classes,
            k_samples=args.hn_k_samples,
            hard_ratio=0.0,
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=batch_sampler,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        )
        logger.info("Hard Negative Mining: ON (P=%d, K=%d)", args.hn_p_classes, args.hn_k_samples)
    elif args.balance_classes:
        class_balanced_sampler = ClassBalancedSampler(
            labels=train_dataset.labels,
            samples_per_class=args.balance_samples_per_class,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=class_balanced_sampler,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )
        logger.info("ClassBalancedSampler: ON (%d samples/class/epoch, "
                     "epoch_size=%d)", args.balance_samples_per_class,
                     len(class_balanced_sampler))
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    model = SpeakerMetricLearner(
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        pretrained_model=args.pretrained_model,
        freeze_feature_extractor=args.freeze_feature_extractor,
        freeze_transformer_layers=args.freeze_transformer_layers,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        num_subcenters=args.num_subcenters,
    ).to(device)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing: ON")

    # Transfer learning
    if args.init_weights is not None:
        ckpt_path = args.init_weights
        if not os.path.isfile(ckpt_path):
            logger.error("init_weights not found: %s", ckpt_path)
            sys.exit(1)
        logger.info("Loading initial weights: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        skipped = [k for k in state_dict if k not in filtered]
        model.load_state_dict(filtered, strict=False)
        logger.info("  Loaded %d / %d params (skipped: %s)", len(filtered), len(state_dict), skipped)

    if args.compile:
        try:
            model = torch.compile(model)
            logger.info("torch.compile: ON")
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %s / %s (%.1f%%)", f"{trainable:,}", f"{total_p:,}",
                100 * trainable / total_p)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs

    def lr_lambda(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return max((epoch_idx + 1) / warmup_epochs, 0.01)
        progress = (epoch_idx - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-6 / args.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    accum_steps = args.grad_accum_steps
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start_epoch = 1

    # --- Resume from checkpoint ---
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            logger.error("Resume checkpoint not found: %s", args.resume)
            sys.exit(1)
        logger.info("Resuming from: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "val_acc" in ckpt and ckpt["val_acc"] is not None:
            best_val_acc = ckpt["val_acc"]
        history_path = os.path.join(args.output_dir, "history.json")
        if os.path.isfile(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        logger.info("  Resumed: epoch=%d, best_val_acc=%.4f", start_epoch, best_val_acc)

    use_file_log = args.log_file is not None
    disable_tqdm = use_file_log
    total_steps = len(train_loader)

    logger.info("Training: epochs %d-%d, %d steps/epoch, batch=%d, lr=%s",
                start_epoch, args.epochs, total_steps, args.batch_size, args.lr)
    logger.info("  ArcFace scale=%.1f, target_margin=%.2f", args.arcface_scale, args.arcface_margin)
    if margin_schedule:
        logger.info("  Margin schedule (%d ep): %s", args.schedule_epochs, margin_schedule)
    if scale_schedule:
        logger.info("  Scale schedule (%d ep): %s", args.schedule_epochs, scale_schedule)
    logger.info("=" * 80)

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        start_time = time.time()

        # Margin / scale scheduling
        if margin_schedule is not None:
            current_margin = _interpolate_schedule(margin_schedule, epoch, args.schedule_epochs)
        elif args.margin_warmup_epochs > 0:
            current_margin = args.arcface_margin * min(1.0, epoch / args.margin_warmup_epochs)
        else:
            current_margin = args.arcface_margin

        current_scale = (
            _interpolate_schedule(scale_schedule, epoch, args.schedule_epochs)
            if scale_schedule is not None
            else args.arcface_scale
        )

        model.arcface.set_margin(current_margin)
        model.arcface.scale = float(current_scale)

        # Hard negative ratio scheduling
        current_hard_ratio = 0.0
        if args.hard_negative and batch_sampler is not None:
            hn_start = args.hn_start_epoch
            hn_warmup = args.hn_warmup_epochs
            if epoch < hn_start:
                current_hard_ratio = 0.0
            elif hn_warmup > 0 and epoch < hn_start + hn_warmup:
                current_hard_ratio = args.hn_hard_ratio * (epoch - hn_start) / hn_warmup
            else:
                current_hard_ratio = args.hn_hard_ratio
            batch_sampler.hard_ratio = current_hard_ratio

            if current_hard_ratio > 0 and (
                epoch % args.hn_update_interval == 0
                or batch_sampler.similarity_matrix is None
            ):
                sim_matrix = compute_class_similarity(model)
                batch_sampler.update_similarity(sim_matrix)
                logger.info("[Epoch %d] Updated hard negative similarity", epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("[Epoch %d/%d] lr=%.2e, margin=%.4f, scale=%.2f, hard_ratio=%.4f",
                    epoch, args.epochs, current_lr, current_margin, current_scale, current_hard_ratio)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=disable_tqdm)
        grad_norm = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, 1):
            global_step += 1
            if args.max_steps and global_step > args.max_steps:
                break

            input_values = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_values, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs["logits"], labels)
                if accum_steps > 1:
                    loss = loss / accum_steps

            scaler.scale(loss).backward()

            is_accum_step = (step % accum_steps == 0) or (step == total_steps)
            if is_accum_step:
                scaler.unscale_(optimizer)
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_norm = _gn.item() if torch.is_tensor(_gn) else float(_gn)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * labels.size(0) * accum_steps
            preds = outputs["logits"].argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

            if step % args.log_interval == 0:
                avg = epoch_loss / epoch_total
                acc = epoch_correct / epoch_total

                with torch.no_grad():
                    logits = outputs["logits"].detach().float()
                    logit_mean = logits.mean().item()
                    logit_std = logits.std().item()
                    logit_max = logits.max().item()
                    logit_min = logits.min().item()
                    bs = labels.size(0)
                    correct_logits = logits[torch.arange(bs, device=device), labels]
                    wrong_mask = torch.ones_like(logits, dtype=torch.bool)
                    wrong_mask[torch.arange(bs, device=device), labels] = False
                    max_wrong_logits = logits.masked_fill(~wrong_mask, -1e9).max(dim=1).values
                    margin_gap = (correct_logits - max_wrong_logits).mean().item()
                    top5_preds = logits.topk(min(5, logits.size(1)), dim=1).indices
                    top5_correct = (top5_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()
                    emb = outputs["embeddings"].detach().float()
                    _W = F.normalize(model.arcface.weight.data, p=2, dim=1)
                    _K = model.arcface.num_subcenters
                    _C = model.arcface.num_classes
                    _W = _W.view(_C, _K, -1)  # (C, K, D)
                    _sub = _W[labels]  # (B, K, D)
                    _cos_all = (emb.unsqueeze(1) * _sub).sum(dim=2)  # (B, K)
                    cos_correct = _cos_all.max(dim=1).values.mean().item()
                    del logits, emb, _W, _sub, _cos_all

                gn = grad_norm
                if disable_tqdm:
                    logger.info(
                        "  [E%d] %d/%d | loss=%.4f acc=%.4f top5=%.4f | "
                        "grad=%.4f | logit[%.1f±%.1f, %.1f~%.1f] gap=%.2f | "
                        "cos_correct=%.4f",
                        epoch, step, total_steps, avg, acc, top5_correct,
                        gn, logit_mean, logit_std, logit_min, logit_max,
                        margin_gap, cos_correct,
                    )
                else:
                    pbar.set_postfix(loss=f"{avg:.4f}", acc=f"{acc:.4f}",
                                     gap=f"{margin_gap:.2f}", cos=f"{cos_correct:.4f}")

            del outputs, loss, input_values, labels, attention_mask

            if args.max_steps and global_step >= args.max_steps:
                break

        del pbar
        if device.type == "cuda":
            torch.cuda.empty_cache()

        scheduler.step()
        elapsed = time.time() - start_time
        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        do_val = (epoch % args.val_interval == 0) or (args.max_steps and global_step >= args.max_steps)
        val_metric = None  # best model 判定に使う指標
        if do_val:
            history["train_loss"].append(train_loss)
            if args.split_by_record:
                val_id2label = val_dataset.id2label
                emb_metrics = evaluate_embeddings(
                    model, val_loader, val_id2label, device, use_amp=use_amp,
                )
                val_acc = emb_metrics["record_acc"]
                val_loss = 0.0
                history["val_loss"].append(None)
                history["val_acc"].append(val_acc)
                if "val_record_acc_macro" not in history:
                    history["val_record_acc_macro"] = []
                    history["val_intra_sim"] = []
                    history["val_inter_sim"] = []
                    history["val_sim_gap"] = []
                history["val_record_acc_macro"].append(emb_metrics["record_acc_macro"])
                history["val_intra_sim"].append(emb_metrics["intra_sim"])
                history["val_inter_sim"].append(emb_metrics["inter_sim"])
                history["val_sim_gap"].append(emb_metrics["sim_gap"])
                val_metric = val_acc
                logger.info(
                    "Epoch %d/%d (%.1fs) | Train: loss=%.4f acc=%.4f | "
                    "Val(emb %d recs): rec_acc=%.4f macro=%.4f intra=%.4f inter=%.4f gap=%.4f",
                    epoch, args.epochs, elapsed, train_loss, train_acc,
                    emb_metrics["n_records"], val_acc, emb_metrics["record_acc_macro"],
                    emb_metrics["intra_sim"], emb_metrics["inter_sim"],
                    emb_metrics["sim_gap"],
                )
            else:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                val_metric = val_acc
                logger.info("Epoch %d/%d (%.1fs) | Train: loss=%.4f acc=%.4f | Val: loss=%.4f acc=%.4f",
                            epoch, args.epochs, elapsed, train_loss, train_acc, val_loss, val_acc)
        else:
            history["train_loss"].append(train_loss)
            logger.info("Epoch %d/%d (%.1fs) | Train: loss=%.4f acc=%.4f | (val skipped)",
                        epoch, args.epochs, elapsed, train_loss, train_acc)

        # --- チェックポイント保存 ---
        _ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc if do_val else None,
            "val_loss": val_loss if do_val else None,
            "train_loss": train_loss,
            "num_classes": num_classes,
            "embedding_dim": args.embedding_dim,
            "pretrained_model": args.pretrained_model,
            "args": vars(args),
        }

        if do_val and val_metric is not None and val_metric > best_val_acc:
            best_val_acc = val_metric
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(_ckpt_payload, best_path)
            logger.info("  -> Best model saved (val_acc=%.4f)", val_metric)

        # 定期チェックポイント (N エポックごと)
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            periodic_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:03d}.pt")
            torch.save(_ckpt_payload, periodic_path)
            logger.info("  -> Periodic checkpoint saved: %s", periodic_path)

        # 最新チェックポイント (毎エポック上書き。途中再開用)
        latest_path = os.path.join(args.output_dir, "latest_model.pt")
        torch.save(_ckpt_payload, latest_path)

        # history.json も毎エポック更新 (途中クラッシュでもログが残る)
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        for h in logging.getLogger().handlers:
            h.flush()

        if args.max_steps and global_step >= args.max_steps:
            logger.info("Reached max_steps=%d; stopping.", args.max_steps)
            break

    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "embedding_dim": args.embedding_dim,
        "pretrained_model": args.pretrained_model,
        "args": vars(args),
    }, final_path)

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 80)
    logger.info("Training complete! Best val acc: %.4f", best_val_acc)
    logger.info("Saved to: %s", args.output_dir)


if __name__ == "__main__":
    train()
