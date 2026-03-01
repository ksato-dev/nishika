"""WavLM + Attentive Statistics Pooling + SubCenter ArcFace 話者メトリック学習モデル。

Pipeline:
  Audio Waveform (16 kHz)
    → WavLM-base-plus (frozen feature extractor + lower layers)
    → Attentive Statistics Pooling (attention-weighted mean & std)
    → Projection Head (Linear → GELU → Dropout → Linear)
    → L2 Normalize → 192-dim Embedding
    → SubCenter ArcFace Head (学習時のみ)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, Wav2Vec2Model, AutoModel


# ---------------------------------------------------------------------------
# Attentive Statistics Pooling
# ---------------------------------------------------------------------------

class AttentiveStatisticsPooling(nn.Module):
    """Attention-weighted mean + std pooling (ECAPA-TDNN style).

    Frame-level hidden states から attention weight を計算し、
    加重平均と加重標準偏差を concat して utterance-level 表現を作る。
    出力次元 = 2 * input_dim。
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H)
            mask: (B, T) — True for valid frames, False for padding
        Returns:
            (B, 2*H)
        """
        # (B, T, 1)
        attn_scores = self.attention(hidden_states)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~mask.unsqueeze(-1), torch.finfo(attn_scores.dtype).min
            )
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)

        # Weighted mean
        mean = (attn_weights * hidden_states).sum(dim=1)  # (B, H)
        # Weighted std
        var = (attn_weights * (hidden_states - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        std = (var.clamp(min=1e-7)).sqrt()  # (B, H)

        return torch.cat([mean, std], dim=1)  # (B, 2*H)


# ---------------------------------------------------------------------------
# SubCenter ArcFace
# ---------------------------------------------------------------------------

class SubCenterArcFaceHead(nn.Module):
    """SubCenter ArcFace — 各クラスに K 個のサブセンターを持つ ArcFace。

    クラス内のモード（話し方のバリエーション等）を複数サブセンターで吸収し、
    最も近いサブセンターの cosine に対して angular margin を適用する。
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 scale: float = 30.0, margin: float = 0.5,
                 num_subcenters: int = 3):
        super().__init__()
        self.scale = scale
        self.target_margin = margin
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters

        # (num_classes * K, embedding_dim) — K sub-centers per class
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes * num_subcenters, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)
        self.set_margin(margin)

    def set_margin(self, margin: float):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = embeddings.float()
        # (C*K, D) → normalize
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        # (B, C*K) cosine similarity with all sub-centers
        cosine_all = F.linear(embeddings, normalized_weight)
        # Reshape → (B, C, K) → take max over sub-centers → (B, C)
        B = cosine_all.size(0)
        cosine_all = cosine_all.view(B, self.num_classes, self.num_subcenters)
        cosine = cosine_all.max(dim=2).values  # (B, C)

        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SpeakerMetricLearner(nn.Module):
    """WavLM backbone + Attentive Stats Pooling + SubCenter ArcFace。

    推論時は extract_embedding() で L2 正規化済み 192 次元ベクトルを取得し、
    ArcFace ヘッドは使わない。
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 192,
        pretrained_model: str = "microsoft/wavlm-base-plus",
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 8,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.5,
        num_subcenters: int = 3,
        asp_bottleneck: int = 128,
    ):
        super().__init__()

        # WavLM / Wav2Vec2 backbone
        if "wavlm" in pretrained_model.lower():
            self.backbone = WavLMModel.from_pretrained(pretrained_model)
        else:
            self.backbone = Wav2Vec2Model.from_pretrained(pretrained_model)
        hidden_size = self.backbone.config.hidden_size

        if freeze_feature_extractor:
            self.backbone.feature_extractor._freeze_parameters()

        if freeze_transformer_layers > 0:
            for i, layer in enumerate(self.backbone.encoder.layers):
                if i < freeze_transformer_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Attentive Statistics Pooling: (B, T, H) → (B, 2*H)
        self.pooling = AttentiveStatisticsPooling(hidden_size, asp_bottleneck)

        # Projection: 2*H → embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, embedding_dim),
        )

        self.arcface = SubCenterArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
            num_subcenters=num_subcenters,
        )

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def enable_gradient_checkpointing(self):
        self.backbone.gradient_checkpointing_enable()

    def _compute_enc_mask(self, attention_mask, enc_len):
        """入力の attention_mask から encoder 出力用フレームマスクを算出。"""
        if attention_mask is None:
            return None
        in_lengths = attention_mask.long().sum(dim=1)
        # feature extractor stride ≈ 320 samples
        out_lengths = (in_lengths.float() / 320.0).ceil().long().clamp(min=1, max=enc_len)
        mask = torch.arange(enc_len, device=attention_mask.device).unsqueeze(0) < out_lengths.unsqueeze(1)
        return mask  # (B, T_enc) bool

    def extract_embedding(self, input_values, attention_mask=None):
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T_enc, H)
        enc_mask = self._compute_enc_mask(attention_mask, hidden_states.size(1))
        pooled = self.pooling(hidden_states, enc_mask)  # (B, 2*H)
        embeddings = self.projector(pooled)  # (B, embedding_dim)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, input_values, attention_mask=None, labels=None):
        embeddings = self.extract_embedding(input_values, attention_mask=attention_mask)
        logits = self.arcface(embeddings, labels)
        return {"embeddings": embeddings, "logits": logits}
