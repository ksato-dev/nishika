"""話者メトリック学習用データセット。

data/<class_name>/*.wav 構成のフォルダから読み込む。
class_name = "<audio_id>__<speaker>" (レコードごとに独立したクラス)
"""

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.utils.rnn as rnn_utils
import torchaudio
from torch.utils.data import Dataset

TARGET_SR = 16_000


@lru_cache(maxsize=64)
def _get_resampler(orig_freq: int, new_freq: int):
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)


# ---------------------------------------------------------------------------
# Waveform augmentations
# ---------------------------------------------------------------------------

class WaveformAugmentor:
    """確率的な波形オーグメンテーション群。"""

    def __init__(
        self,
        sr: int = TARGET_SR,
        noise_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (10, 40),
        gain_prob: float = 0.5,
        gain_db_range: Tuple[float, float] = (-6, 6),
        time_shift_prob: float = 0.3,
        time_shift_max_ratio: float = 0.1,
        speed_prob: float = 0.3,
        speed_range: Tuple[float, float] = (0.9, 1.1),
        time_mask_prob: float = 0.3,
        time_mask_max_ratio: float = 0.1,
        time_mask_num: int = 2,
        polarity_prob: float = 0.2,
        pitch_prob: float = 0.0,
        pitch_range_semitones: float = 2.0,
        reverb_prob: float = 0.0,
        reverb_decay: float = 4.0,
        reverb_room_size: float = 0.3,
        reverb_wet: float = 0.3,
    ):
        self.sr = sr
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.gain_prob = gain_prob
        self.gain_db_range = gain_db_range
        self.time_shift_prob = time_shift_prob
        self.time_shift_max_ratio = time_shift_max_ratio
        self.speed_prob = speed_prob
        self.speed_range = speed_range
        self.time_mask_prob = time_mask_prob
        self.time_mask_max_ratio = time_mask_max_ratio
        self.time_mask_num = time_mask_num
        self.polarity_prob = polarity_prob
        self.pitch_prob = pitch_prob
        self.pitch_range_semitones = pitch_range_semitones
        self.reverb_prob = reverb_prob
        self.reverb_decay = reverb_decay
        self.reverb_room_size = reverb_room_size
        self.reverb_wet = reverb_wet

    @staticmethod
    def _add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
        noise = np.random.randn(len(audio)).astype(np.float32)
        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0 or audio_power == 0:
            return audio
        scale = np.sqrt(audio_power / (noise_power * 10 ** (snr_db / 10)))
        return audio + noise * scale

    @staticmethod
    def _gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
        return audio * (10 ** (gain_db / 20))

    @staticmethod
    def _time_shift(audio: np.ndarray, max_shift_ratio: float) -> np.ndarray:
        max_shift = int(len(audio) * max_shift_ratio)
        if max_shift == 0:
            return audio
        shift = random.randint(-max_shift, max_shift)
        return np.roll(audio, shift)

    @staticmethod
    def _speed_perturb(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        new_sr = int(sr * factor)
        if new_sr == sr:
            return audio
        resampler = _get_resampler(new_sr, sr)
        return resampler(audio_t).squeeze(0).numpy()

    @staticmethod
    def _time_mask(audio: np.ndarray, num_masks: int, max_mask_ratio: float) -> np.ndarray:
        result = audio.copy()
        for _ in range(num_masks):
            mask_len = random.randint(1, max(1, int(len(audio) * max_mask_ratio)))
            start = random.randint(0, max(0, len(audio) - mask_len))
            result[start:start + mask_len] = 0.0
        return result

    @staticmethod
    def _polarity_inversion(audio: np.ndarray) -> np.ndarray:
        return -audio

    @staticmethod
    def _pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        factor = 2.0 ** (semitones / 12.0)
        new_sr = int(sr * factor)
        if new_sr == sr:
            return audio
        orig_len = len(audio)
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        resampler = _get_resampler(new_sr, sr)
        result = resampler(audio_t).squeeze(0).numpy()
        if len(result) > orig_len:
            result = result[:orig_len]
        elif len(result) < orig_len:
            result = np.pad(result, (0, orig_len - len(result)), mode="constant")
        return result.astype(np.float32)

    @staticmethod
    def _reverb(audio: np.ndarray, sr: int, decay: float,
                room_size: float, wet: float) -> np.ndarray:
        ir_len = max(int(sr * room_size), 1)
        t = np.linspace(0, 1, ir_len, dtype=np.float32)
        ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-decay * t * 10)
        ir[0] = 1.0
        ir = ir / (np.linalg.norm(ir) + 1e-9)
        n = len(audio) + ir_len - 1
        fft_n = 1
        while fft_n < n:
            fft_n *= 2
        audio_fft = np.fft.rfft(audio, fft_n)
        ir_fft = np.fft.rfft(ir, fft_n)
        result = np.fft.irfft(audio_fft * ir_fft, fft_n)[:len(audio)]
        return ((1.0 - wet) * audio + wet * result.astype(np.float32)).astype(np.float32)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if random.random() < self.noise_prob:
            audio = self._add_noise(audio, random.uniform(*self.noise_snr_range))
        if random.random() < self.gain_prob:
            audio = self._gain(audio, random.uniform(*self.gain_db_range))
        if random.random() < self.time_shift_prob:
            audio = self._time_shift(audio, self.time_shift_max_ratio)
        if random.random() < self.speed_prob:
            audio = self._speed_perturb(audio, self.sr, random.uniform(*self.speed_range))
        if random.random() < self.pitch_prob:
            audio = self._pitch_shift(audio, self.sr,
                                      random.uniform(-self.pitch_range_semitones,
                                                     self.pitch_range_semitones))
        if random.random() < self.reverb_prob:
            audio = self._reverb(audio, self.sr, self.reverb_decay,
                                 self.reverb_room_size, self.reverb_wet)
        if random.random() < self.time_mask_prob:
            audio = self._time_mask(audio, self.time_mask_num, self.time_mask_max_ratio)
        if random.random() < self.polarity_prob:
            audio = self._polarity_inversion(audio)
        return audio


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path, target_sr=TARGET_SR, max_length=None, random_crop=True):
    path = Path(path)
    if path.suffix.lower() == ".wav":
        audio, sr = sf.read(path, dtype="float32")
    else:
        audio_tensor, sr = torchaudio.load(path)
        audio = audio_tensor.numpy()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        audio = audio.astype("float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = _get_resampler(sr, target_sr)
        audio = resampler(audio_tensor).squeeze(0).numpy()

    if max_length is not None:
        if len(audio) > max_length:
            if random_crop:
                start = random.randint(0, len(audio) - max_length)
            else:
                start = (len(audio) - max_length) // 2
            audio = audio[start:start + max_length]
        elif len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)), mode="constant")
    return audio


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

def build_label_map(data_dir):
    data_path = Path(data_dir)
    class_dirs = sorted([
        d.name for d in data_path.iterdir()
        if d.is_dir()
    ])
    label2id = {name: idx for idx, name in enumerate(class_dirs)}
    return label2id, class_dirs


# ---------------------------------------------------------------------------
# Balancing helpers
# ---------------------------------------------------------------------------

def _count_per_class(labels) -> dict[int, int]:
    """ラベルリストからクラスごとのサンプル数を返す。"""
    from collections import Counter
    return dict(Counter(labels))


def _print_distribution(counts: dict[int, int], prefix: str = ""):
    """クラスごとサンプル数の分布統計を print する。"""
    vals = np.array(list(counts.values()))
    n_cls = len(vals)
    if n_cls == 0:
        print(f"{prefix}(empty)")
        return
    print(f"{prefix}{n_cls} classes, {vals.sum()} samples | "
          f"min={vals.min()} max={vals.max()} mean={vals.mean():.1f} "
          f"median={np.median(vals):.0f} | "
          f"<10:{(vals<10).sum()} <20:{(vals<20).sum()} <50:{(vals<50).sum()} "
          f">100:{(vals>100).sum()} >200:{(vals>200).sum()}")


def _downsample(files, labels, max_per_class, rng):
    """クラスごとのサンプル数を max_per_class 以下に間引く。"""
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, lab in enumerate(labels):
        class_indices[lab].append(i)

    before_counts = {lab: len(idxs) for lab, idxs in class_indices.items()}
    n_capped = sum(1 for c in before_counts.values() if c > max_per_class)

    selected = []
    for lab, idxs in class_indices.items():
        if len(idxs) <= max_per_class:
            selected.extend(idxs)
        else:
            selected.extend(rng.sample(idxs, max_per_class))
    selected.sort()

    new_files = [files[i] for i in selected]
    new_labels = [labels[i] for i in selected]
    after_counts = _count_per_class(new_labels)

    print(f"  Downsample cap={max_per_class}: "
          f"{len(files)} -> {len(new_files)} samples, "
          f"{n_capped}/{len(before_counts)} classes capped")
    _print_distribution(before_counts, "    Before: ")
    _print_distribution(after_counts,  "    After:  ")

    return new_files, new_labels


class ClassBalancedSampler:
    """クラスバランスサンプラー。

    各エポックで全クラスから均等にサンプルする。
    少数クラスはオーバーサンプリング、多数クラスはアンダーサンプリング。
    1 エポック = num_classes × samples_per_class サンプル。
    """

    def __init__(self, labels, samples_per_class: int = 20, seed: int = 42):
        self.labels = np.asarray(labels)
        self.samples_per_class = samples_per_class

        self.class_to_indices: dict[int, list[int]] = {}
        for idx, lab in enumerate(self.labels):
            self.class_to_indices.setdefault(int(lab), []).append(idx)
        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        self._rng = random.Random(seed)

        actual_counts = np.array([len(self.class_to_indices[c]) for c in self.classes])
        n_over = int((actual_counts >= samples_per_class).sum())
        n_under = int((actual_counts < samples_per_class).sum())
        print(f"  ClassBalancedSampler: {self.num_classes} classes × "
              f"{samples_per_class} samples = {len(self)} / epoch | "
              f"oversample={n_under} classes, undersample={n_over} classes")

    def __iter__(self):
        indices = []
        for cls in self.classes:
            pool = self.class_to_indices[cls]
            if len(pool) >= self.samples_per_class:
                indices.extend(self._rng.sample(pool, self.samples_per_class))
            else:
                indices.extend(self._rng.choices(pool, k=self.samples_per_class))
        self._rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_classes * self.samples_per_class


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _extract_record_id(class_name: str, sep: str = "__") -> str:
    """クラス名 '<audio_id>__<speaker>' から audio_id を抽出する。"""
    parts = class_name.split(sep)
    return parts[0] if len(parts) >= 2 else class_name


class SpeakerClipDataset(Dataset):
    """話者クリップデータセット。

    preload=True で全波形を共有メモリの flat buffer に載せて
    DataLoader の worker 間コピーを高速化する。

    split_by_record=True のとき、レコード (audio_id) 単位で train/val を分割する。
    val 側のクラスは train に一切含まれないため、未知話者での汎化性能を評価できる。

    max_samples_per_class: 各クラスのサンプル数上限。超過分はランダムに間引く。
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        validation_ratio: float = 0.2,
        max_length: Optional[int] = None,
        label2id: Optional[dict] = None,
        seed: int = 42,
        augmentor: Optional[WaveformAugmentor] = None,
        preload: bool = True,
        max_samples_per_class: Optional[int] = None,
        split_by_record: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.split = split
        self.augmentor = augmentor if split == "train" else None

        if label2id is not None:
            self.label2id = label2id
        else:
            self.label2id, _ = build_label_map(data_dir)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)

        # --- 全ファイル列挙 ---
        all_files = []
        all_labels = []
        all_class_names = []
        for class_name, class_id in self.label2id.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for wav_file in sorted(class_dir.glob("*.wav")):
                all_files.append(str(wav_file))
                all_labels.append(class_id)
                all_class_names.append(class_name)

        rng = random.Random(seed)

        if split_by_record:
            # --- レコード (audio_id) 単位で split ---
            record_ids = sorted(set(
                _extract_record_id(cn) for cn in self.label2id.keys()
            ))
            rng.shuffle(record_ids)
            split_idx = int(len(record_ids) * (1 - validation_ratio))
            if split == "train":
                split_records = set(record_ids[:split_idx])
            else:
                split_records = set(record_ids[split_idx:])

            selected = [
                i for i, cn in enumerate(all_class_names)
                if _extract_record_id(cn) in split_records
            ]

            n_train_records = split_idx
            n_val_records = len(record_ids) - split_idx
            print(f"[{split}] Record-level split: "
                  f"{n_train_records} train / {n_val_records} val records "
                  f"(total {len(record_ids)})")
        else:
            # --- サンプル単位で shuffle split (従来方式) ---
            indices = list(range(len(all_files)))
            rng.shuffle(indices)
            split_idx = int(len(indices) * (1 - validation_ratio))
            if split == "train":
                selected = indices[:split_idx]
            else:
                selected = indices[split_idx:]

        self.files = [all_files[i] for i in selected]
        self.labels = [all_labels[i] for i in selected]

        # --- 分布表示 ---
        _print_distribution(_count_per_class(self.labels), f"[{split}] Distribution: ")

        # --- ダウンサンプリング ---
        if max_samples_per_class is not None and max_samples_per_class > 0 and split == "train":
            self.files, self.labels = _downsample(
                self.files, self.labels, max_samples_per_class, rng,
            )

        self._flat_audio: Optional[torch.Tensor] = None
        self._offsets: Optional[np.ndarray] = None
        self._lengths: Optional[np.ndarray] = None

        if preload:
            chunks = []
            lengths = []
            for f in self.files:
                a = load_audio(f, max_length=None)
                chunks.append(a)
                lengths.append(len(a))
            flat_np = np.concatenate(chunks)
            self._flat_audio = torch.from_numpy(flat_np).share_memory_()
            del flat_np
            self._lengths = np.array(lengths, dtype=np.int64)
            offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
            np.cumsum(self._lengths, out=offsets[1:])
            self._offsets = offsets
            total_mb = self._flat_audio.nbytes / (1024 * 1024)
            print(f"[{split}] {len(self.files)} samples, {self.num_classes} classes "
                  f"(preloaded {total_mb:.1f} MB)")
        else:
            print(f"[{split}] {len(self.files)} samples, {self.num_classes} classes")

    def __len__(self):
        return len(self.files)

    def _get_audio_from_cache(self, idx):
        start = self._offsets[idx]
        end = self._offsets[idx + 1]
        return self._flat_audio[start:end].numpy().copy()

    def __getitem__(self, idx):
        if self._flat_audio is not None:
            audio = self._get_audio_from_cache(idx)
            if self.max_length is not None:
                if len(audio) > self.max_length:
                    start = random.randint(0, len(audio) - self.max_length)
                    audio = audio[start:start + self.max_length]
                elif len(audio) < self.max_length:
                    audio = np.pad(audio, (0, self.max_length - len(audio)), mode="constant")
        else:
            audio = load_audio(self.files[idx], max_length=self.max_length)
        if self.augmentor is not None:
            audio = self.augmentor(audio)
        return {
            "input_values": torch.from_numpy(audio).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "length": len(audio),
        }


def collate_fn(batch):
    """可変長バッチを pad_sequence でパディング。"""
    labels = torch.stack([item["label"] for item in batch])
    waveforms = [item["input_values"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    input_values = rnn_utils.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    max_len = input_values.size(1)
    attention_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "input_values": input_values,
        "attention_mask": attention_mask.long(),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Hard Negative Mining -- PK batch sampler
# ---------------------------------------------------------------------------

class HardNegativeBatchSampler:
    """P classes × K samples のミニバッチを構成するサンプラー。

    ArcFace 重みベクトルからクラス間類似度を計算し、hard_ratio の割合で
    紛らわしいクラスを同一バッチに集める。
    """

    def __init__(self, labels, p_classes=4, k_samples=2, hard_ratio=0.5):
        self.labels = np.asarray(labels)
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.hard_ratio = hard_ratio

        self.class_to_indices = {}
        for idx, lab in enumerate(self.labels):
            self.class_to_indices.setdefault(int(lab), []).append(idx)
        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        self._class_to_pos = {c: i for i, c in enumerate(self.classes)}
        self.similarity_matrix = None

    def update_similarity(self, similarity_matrix):
        self.similarity_matrix = similarity_matrix

    def __iter__(self):
        total_samples = sum(len(v) for v in self.class_to_indices.values())
        num_batches = total_samples // (self.p_classes * self.k_samples)

        for _ in range(num_batches):
            if self.similarity_matrix is not None and random.random() < self.hard_ratio:
                anchor_cls = random.choice(self.classes)
                anchor_pos = self._class_to_pos[anchor_cls]
                sims = self.similarity_matrix[anchor_pos].copy()
                sims[anchor_pos] = -np.inf
                top_pos = np.argsort(sims)[::-1][:self.p_classes - 1]
                selected = [anchor_cls] + [self.classes[p] for p in top_pos]
            else:
                p = min(self.p_classes, self.num_classes)
                selected = random.sample(self.classes, p)

            batch = []
            for cls in selected:
                pool = self.class_to_indices[cls]
                if len(pool) >= self.k_samples:
                    batch.extend(random.sample(pool, self.k_samples))
                else:
                    batch.extend(random.choices(pool, k=self.k_samples))
            yield batch

    def __len__(self):
        total_samples = sum(len(v) for v in self.class_to_indices.values())
        return total_samples // (self.p_classes * self.k_samples)
