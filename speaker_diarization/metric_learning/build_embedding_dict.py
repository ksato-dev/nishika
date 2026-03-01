"""学習済みモデルから話者ごとのリファレンス埋め込み辞書を構築する。

使い方:
  python metric_learning/build_embedding_dict.py \
    --data_dir ./metric_learning/data \
    --checkpoint ./metric_learning/checkpoints/best_model.pt \
    --output ./metric_learning/checkpoints/speaker_embeddings.npz
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeakerClipDataset, collate_fn, build_label_map, load_audio
from model import SpeakerMetricLearner


def load_model(checkpoint_path, device, data_dir=None):
    """チェックポイントからモデルをロードする。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if data_dir:
        label2id, class_names = build_label_map(data_dir)
    else:
        label_map_dir = str(Path(checkpoint_path).parent)
        lm_path = os.path.join(label_map_dir, "label_map.json")
        with open(lm_path, encoding="utf-8") as f:
            lm = json.load(f)
        label2id = lm["label2id"]
        class_names = lm["class_names"]

    num_classes = ckpt.get("num_classes", len(label2id))
    embedding_dim = ckpt.get("embedding_dim", 192)
    pretrained_model = ckpt.get("pretrained_model", "microsoft/wavlm-base-plus")

    args_dict = ckpt.get("args", {})
    model = SpeakerMetricLearner(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained_model=pretrained_model,
        freeze_feature_extractor=args_dict.get("freeze_feature_extractor", True),
        freeze_transformer_layers=args_dict.get("freeze_transformer_layers", 8),
        num_subcenters=args_dict.get("num_subcenters", 3),
    )

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, label2id, class_names


@torch.inference_mode()
def extract_all_embeddings(model, data_dir, label2id, device, batch_size=32):
    """全サンプルの埋め込みを抽出する。"""
    dataset = SpeakerClipDataset(
        data_dir=data_dir, split="train", validation_ratio=0.0,
        label2id=label2id, preload=True,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    all_embeddings = []
    all_labels = []
    for batch in tqdm(loader, desc="Extracting embeddings"):
        input_values = batch["input_values"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        emb = model.extract_embedding(input_values, attention_mask=attention_mask)
        all_embeddings.append(emb.cpu())
        all_labels.append(batch["labels"])

    return torch.cat(all_embeddings), torch.cat(all_labels)


def aggregate_embeddings(embeddings, labels, class_names, label2id,
                         method="outlier_trimmed", outlier_fraction=0.2,
                         trimmed_fraction=0.1):
    """クラスごとに埋め込みを集約して代表ベクトルを作成する。"""
    id2label = {v: k for k, v in label2id.items()}
    result = {}

    for class_name in class_names:
        class_id = label2id[class_name]
        mask = labels == class_id
        if mask.sum() == 0:
            continue
        class_embs = embeddings[mask]

        if method == "mean":
            center = class_embs.mean(dim=0)
        elif method == "median":
            center = class_embs.median(dim=0).values
        elif method == "trimmed_mean":
            center = _trimmed_mean(class_embs, trimmed_fraction)
        elif method == "outlier_trimmed":
            center = _outlier_trimmed_mean(class_embs, outlier_fraction)
        else:
            center = class_embs.mean(dim=0)

        center = F.normalize(center, p=2, dim=0)
        result[class_name] = center

    return result


def _trimmed_mean(embs, fraction):
    """仮の平均からの距離でソートし上下 fraction を除外して平均。"""
    if len(embs) <= 2:
        return embs.mean(dim=0)
    mean = embs.mean(dim=0, keepdim=True)
    cos_dist = 1 - F.cosine_similarity(embs, mean)
    n_trim = int(len(embs) * fraction)
    if n_trim == 0:
        return embs.mean(dim=0)
    order = cos_dist.argsort()
    kept = order[:len(embs) - n_trim]
    return embs[kept].mean(dim=0)


def _outlier_trimmed_mean(embs, fraction):
    """仮の平均からコサイン類似度が低い下位 fraction を除外して平均。"""
    if len(embs) <= 2:
        return embs.mean(dim=0)
    mean = embs.mean(dim=0, keepdim=True)
    cos_sim = F.cosine_similarity(embs, mean)
    n_drop = int(len(embs) * fraction)
    if n_drop == 0:
        return embs.mean(dim=0)
    order = cos_sim.argsort(descending=True)
    kept = order[:len(embs) - n_drop]
    return embs[kept].mean(dim=0)


def main():
    p = argparse.ArgumentParser(description="Build speaker embedding dictionary")
    p.add_argument("--data_dir", type=str, required=True, default="./metric_learning/data")
    p.add_argument("--checkpoint", type=str, default="./metric_learning/checkpoints/best_model.pt")
    p.add_argument("--output", type=str, default="./metric_learning/checkpoints/speaker_embeddings.npz")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--aggregation", type=str, default="outlier_trimmed",
                   choices=["mean", "median", "trimmed_mean", "outlier_trimmed"])
    p.add_argument("--outlier_fraction", type=float, default=0.2)
    p.add_argument("--trimmed_fraction", type=float, default=0.1)
    args = p.parse_args()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    print("Loading model...")
    model, label2id, class_names = load_model(args.checkpoint, device, args.data_dir)
    print(f"Model loaded: {len(class_names)} classes, embedding_dim={model.embedding_dim}")

    print("Extracting embeddings...")
    embeddings, labels = extract_all_embeddings(
        model, args.data_dir, label2id, device, batch_size=args.batch_size,
    )
    print(f"Extracted: {embeddings.shape}")

    print(f"Aggregating ({args.aggregation})...")
    result = aggregate_embeddings(
        embeddings, labels, class_names, label2id,
        method=args.aggregation,
        outlier_fraction=args.outlier_fraction,
        trimmed_fraction=args.trimmed_fraction,
    )

    out_names = sorted(result.keys())
    out_embs = torch.stack([result[n] for n in out_names]).numpy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=out_embs, class_names=np.array(out_names))
    print(f"Saved: {out_path} ({len(out_names)} classes, shape={out_embs.shape})")


if __name__ == "__main__":
    main()
