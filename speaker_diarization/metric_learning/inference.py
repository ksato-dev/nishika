"""学習済みモデルで音声ファイルから話者埋め込みを抽出する推論スクリプト。

使い方 (単一ファイル):
  python metric_learning/inference.py \
    --checkpoint ./metric_learning/checkpoints/best_model.pt \
    --audio ./input/train/0HSiCLDz8l/voiceprints/A.wav

使い方 (ディレクトリ):
  python metric_learning/inference.py \
    --checkpoint ./metric_learning/checkpoints/best_model.pt \
    --audio_dir ./input/train/0HSiCLDz8l/voiceprints/ \
    --load_embeddings ./metric_learning/checkpoints/speaker_embeddings.npz
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import load_audio, TARGET_SR
from model import SpeakerMetricLearner


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = ckpt.get("args", {})
    num_classes = ckpt.get("num_classes", args_dict.get("num_classes", 100))
    embedding_dim = ckpt.get("embedding_dim", args_dict.get("embedding_dim", 192))
    pretrained_model = ckpt.get("pretrained_model", args_dict.get("pretrained_model", "microsoft/wavlm-base-plus"))

    model = SpeakerMetricLearner(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained_model=pretrained_model,
        freeze_feature_extractor=True,
        freeze_transformer_layers=12,
        num_subcenters=args_dict.get("num_subcenters", 3),
    )
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def extract_embedding(model, audio_path, device, multi_crop_shift_ms=200.0):
    """音声ファイルから L2 正規化済み埋め込みを抽出する。

    multi_crop_shift_ms > 0 のとき、center / left-shift / right-shift の
    3 つの切り出しの埋め込みを平均する。
    """
    audio = load_audio(audio_path, random_crop=False)
    shift_samples = int(multi_crop_shift_ms / 1000.0 * TARGET_SR)

    crops = [audio]
    if shift_samples > 0 and len(audio) > shift_samples * 2:
        left = audio[shift_samples:]
        left = np.pad(left, (0, shift_samples), mode="constant")
        crops.append(left)

        right = np.pad(audio, (shift_samples, 0), mode="constant")[:len(audio)]
        crops.append(right)

    embeddings = []
    for crop in crops:
        t = torch.from_numpy(crop).float().unsqueeze(0).to(device)
        emb = model.extract_embedding(t)
        embeddings.append(emb.cpu())

    mean_emb = torch.stack(embeddings).mean(dim=0)
    return F.normalize(mean_emb, p=2, dim=1).squeeze(0)


def load_reference_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = torch.from_numpy(data["embeddings"]).float()
    class_names = list(data["class_names"])
    return embeddings, class_names


def search(query_emb, ref_embeddings, class_names, top_k=5):
    """コサイン類似度で最も近いクラスを検索する。"""
    cos_sim = F.cosine_similarity(query_emb.unsqueeze(0), ref_embeddings)
    top_vals, top_idx = cos_sim.topk(min(top_k, len(class_names)))
    results = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        results.append((class_names[idx], val))
    return results


def main():
    p = argparse.ArgumentParser(description="Speaker embedding inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--audio", type=str, default=None, help="単一音声ファイル")
    p.add_argument("--audio_dir", type=str, default=None, help="音声ファイルのディレクトリ")
    p.add_argument("--load_embeddings", type=str, default=None,
                   help="リファレンス埋め込み (.npz) — 指定時は類似度検索を実行")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--multi_crop_shift_ms", type=float, default=200.0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)
    print("Model loaded.")

    ref_embeddings, ref_names = None, None
    if args.load_embeddings:
        ref_embeddings, ref_names = load_reference_embeddings(args.load_embeddings)
        print(f"Reference: {len(ref_names)} classes")

    audio_files = []
    if args.audio:
        audio_files.append(args.audio)
    if args.audio_dir:
        d = Path(args.audio_dir)
        audio_files.extend(sorted(str(f) for f in d.glob("*.wav")))
        audio_files.extend(sorted(str(f) for f in d.glob("*.mp3")))

    for af in audio_files:
        emb = extract_embedding(model, af, device, args.multi_crop_shift_ms)
        print(f"\n{Path(af).name}: embedding shape={emb.shape}")
        if ref_embeddings is not None:
            results = search(emb, ref_embeddings, ref_names, args.top_k)
            for rank, (name, sim) in enumerate(results, 1):
                print(f"  #{rank} {name}  cos_sim={sim:.4f}")


if __name__ == "__main__":
    main()
