# Speaker Metric Learning

WavLM-base-plus + Attentive Statistics Pooling + SubCenter ArcFace による
話者メトリック学習パイプライン。
話者ダイアリゼーション用のアノテーション付き音声データから話者ごとの埋め込みベクトルを学習する。

## アーキテクチャ

```
Audio Waveform (16 kHz, 可変長)
  |
  v
WavLM-base-plus (microsoft/wavlm-base-plus)
  frozen feature extractor + lower 8 layers
  |
  v
Attentive Statistics Pooling
  attention-weighted mean & std → concat → 2×768 = 1536-dim
  |
  v
Projection Head (Linear 1536→768 → GELU → Dropout → Linear 768→192)
  |
  v
L2 Normalize → 192-dim Embedding
  |
  v
SubCenter ArcFace (K=3 sub-centers per class)
  学習時のみ。推論時は embedding だけ使う
```

### なぜこの構成か

| コンポーネント | 理由 |
|---|---|
| WavLM-base-plus | wav2vec2 より話者特性の抽出に優れた事前学習。話者検証タスクで SOTA 級 |
| Attentive Statistics Pooling | 平均だけでなく分散情報も捉える。発話の「揺らぎ」を表現に含められる |
| 192-dim embedding | 話者検証の標準的な次元数。512-dim より軽量で十分な識別力 |
| SubCenter ArcFace (K=3) | 同一話者の発話スタイル変動（声質・テンション等）を複数サブセンターで吸収 |

## データ設計

```
input/train/<audio_id>/<audio_id>.wav          会議音声 (数分〜数十分)
input/train/<audio_id>/voiceprints/<speaker>.wav 話者ごとの参照音声
input/train_annotation.csv                      発話区間アノテーション
  → audio_id, start_time, end_time, speaker
```

**クラス設計**: `<audio_id>__<speaker>` を 1 クラスとして扱う。
異なるレコードの同一話者 ID (A, B, C, ...) はすべて**別クラス**。

抽出後:
```
metric_learning/data/
  0HSiCLDz8l__A/
    seg_0001.wav      アノテーション区間から切り出し
    seg_0002.wav
    ...
    voiceprint.wav    voiceprints/ からコピー
  0HSiCLDz8l__B/
    ...
```

## ディレクトリ構成

```
metric_learning/
  README.md
  extract_clips.py          アノテーション CSV → 話者クリップ抽出
  model.py                  モデル定義 (WavLM + ASP + SubCenter ArcFace)
  dataset.py                データセット / augmentation / collate / サンプラー
  train.py                  学習スクリプト
  build_embedding_dict.py   話者リファレンス埋め込み辞書の構築
  inference.py              推論・類似度検索
  speaker_stats.py          クラスごとの話者時間統計分析
  data/                     抽出済みクリップ (クラスごとのサブフォルダ)
  checkpoints/              モデル・学習ログ
    best_model.pt           ベスト val 精度のチェックポイント
    latest_model.pt         最新エポック (途中再開用、毎エポック上書き)
    checkpoint_epochNNN.pt  定期チェックポイント (--save_interval)
    final_model.pt          最終エポック
    label_map.json          クラス名 ↔ ID のマッピング
    history.json            学習履歴 (毎エポック更新)
    speaker_embeddings.npz  リファレンス埋め込み辞書
    train_log.txt           学習ログ (リアルタイム監視用)
```

## 使い方

### 0. 環境準備

```bash
# 既存 venv を使う (torch, torchaudio 等インストール済み)
pip install "transformers>=4.40,<5" soundfile
```

### 1. クリップ抽出

```bash
python metric_learning/extract_clips.py \
  --annotation_csv ./input/train_annotation.csv \
  --train_dir ./input/train \
  --output_dir ./metric_learning/data \
  --min_duration 0.5
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--min_duration` | `0.5` | 最小セグメント長 (秒)。これ未満は除外 |
| `--max_duration` | `15.0` | 最大セグメント長 (秒)。超過分はトリム |
| `--no_voiceprints` | `False` | voiceprint wav を含めない |

### 2. 学習

```bash
python metric_learning/train.py \
  --data_dir ./metric_learning/data \
  --output_dir ./metric_learning/checkpoints \
  --epochs 50 \
  --batch_size 24 \
  --lr 3e-4 \
  --warmup_epochs 3 \
  --margin_schedule "0,0.05,0.1,0.15,0.2,0.25,0.3,0.35" \
  --scale_schedule "20,32,40,48" \
  --schedule_epochs 10 \
  --hard_negative \
  --hn_start_epoch 13 \
  --hn_warmup_epochs 5 \
  --log_file ./metric_learning/checkpoints/train_log.txt \
  --gradient_checkpointing
```

#### 学習スケジュール

```
Epoch  1-3   LR warmup (linear 0→lr)
             margin=0, scale=20 からスタート
Epoch  1-10  margin / scale を徐々に引き上げ
             margin: 0 → 0.05 → 0.1 → ... → 0.35
             scale:  20 → 32 → 40 → 48
Epoch 11-12  パラメータ固定で慣らし (margin=0.35, scale=48)
Epoch 13-17  Hard Negative Mining を徐々に導入
             hard_ratio: 0 → 0.1 → ... → 0.5
Epoch 18-50  フル Hard Negative Mining (hard_ratio=0.5)
```

#### 主要オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--pretrained_model` | `microsoft/wavlm-base-plus` | backbone モデル |
| `--embedding_dim` | `192` | 埋め込み次元数 |
| `--num_subcenters` | `3` | SubCenter ArcFace のサブセンター数 |
| `--arcface_scale` | `20.0` | ArcFace scale (初期値。schedule で変更可) |
| `--arcface_margin` | `0.0` | ArcFace margin (初期値。schedule で変更可) |
| `--freeze_transformer_layers` | `8` | フリーズする Transformer 層数 |
| `--warmup_epochs` | `2` | LR linear warmup エポック数 |
| `--margin_schedule` | `None` | カンマ区切り margin スケジュール |
| `--scale_schedule` | `None` | カンマ区切り scale スケジュール |
| `--schedule_epochs` | `10` | margin/scale スケジュールの適用期間 |
| `--gradient_checkpointing` | `False` | activation memory を ~60% 削減 |
| `--grad_accum_steps` | `1` | 勾配累積。実効 batch = batch_size × steps |
| `--init_weights` | `None` | 転移学習用の初期重みパス |

#### Train / Val 分割

| オプション | デフォルト | 説明 |
|---|---|---|
| `--split_by_record` | `True` | レコード (audio_id) 単位で分割。val 側は完全未知の話者 |
| `--no_split_by_record` | — | 従来のサンプル単位 shuffle split に戻す |
| `--val_ratio` | `0.2` | validation の割合 |
| `--val_interval` | `1` | 何エポックごとに validation を実行するか |

`--split_by_record` (デフォルト ON) では、レコード単位でシャッフルし 80/20 に分割する。
val 側のレコードに属するクラス（話者）は train に一切含まれないため、
**未知話者に対する汎化性能**を正しく評価できる。

#### データバランシング

| オプション | デフォルト | 説明 |
|---|---|---|
| `--max_samples_per_class` | `None` | クラスあたりの最大サンプル数。超過分をランダムに間引く |
| `--balance_classes` | `False` | ClassBalancedSampler で各クラスから均等にサンプル |
| `--balance_samples_per_class` | `20` | balance_classes 時の 1 エポックあたりクラスあたりサンプル数 |

※ `--hard_negative` が ON のときはそちらのサンプラーが優先される

#### Hard Negative Mining

| オプション | デフォルト | 説明 |
|---|---|---|
| `--hard_negative` | `False` | 有効化 |
| `--hn_p_classes` | `6` | バッチ内クラス数 (P)。batch = P×K |
| `--hn_k_samples` | `4` | クラスあたりサンプル数 (K) |
| `--hn_hard_ratio` | `0.5` | hard batch の割合 |
| `--hn_start_epoch` | `schedule_epochs+1` | 開始エポック |
| `--hn_warmup_epochs` | `5` | hard_ratio の ramp-up 期間 |
| `--hn_update_interval` | `5` | 類似度行列の更新間隔 (エポック) |

#### オーグメンテーション

デフォルトで有効。`--no_augment` で全て無効化。

| 種類 | 確率 | パラメータ |
|---|---|---|
| ガウシアンノイズ | 0.5 | SNR 10–40 dB |
| ゲイン変動 | 0.5 | -6 to +6 dB |
| タイムシフト | 0.3 | 最大 10% |
| スピード変動 | 0.3 | 0.9x–1.1x |
| タイムマスク | 0.3 | 最大 10%, 2 回 |
| 極性反転 | 0.2 | — |
| ピッチシフト | 0.0 (off) | ±2 semitones |
| リバーブ | 0.0 (off) | decay=4.0, room=0.3, wet=0.3 |

#### チェックポイント

| オプション | デフォルト | 説明 |
|---|---|---|
| `--save_interval` | `5` | 定期チェックポイントの保存間隔 (エポック) |

自動で保存されるファイル:

| ファイル | タイミング | 説明 |
|---|---|---|
| `best_model.pt` | val 精度更新時 | ベストモデル |
| `latest_model.pt` | 毎エポック | 途中再開用 (上書き) |
| `checkpoint_epochNNN.pt` | N エポックごと | 定期スナップショット |
| `final_model.pt` | 学習完了時 | 最終モデル |
| `history.json` | 毎エポック | 学習履歴 |

#### 学習ログの監視

```powershell
Get-Content ./metric_learning/checkpoints/train_log.txt -Wait
```

##### ステップログ (学習中)

```
[E1] 10/275 | loss=6.5781 acc=0.0000 top5=0.0000 | grad=34.3456 | logit[0.9±1.1, -2.3~5.4] gap=-3.83 | cos_correct=0.0439
```

| メトリクス | 意味 |
|---|---|
| `loss` | CrossEntropy loss (epoch 累積平均) |
| `acc` | Top-1 accuracy |
| `top5` | Top-5 accuracy |
| `grad` | gradient norm |
| `logit[mean±std, min~max]` | ArcFace logit の統計量 |
| `gap` | 正解 logit - 最大不正解 logit の平均 |
| `cos_correct` | embedding と正解クラスの最近サブセンターとの cosine 類似度 |

##### エポック終了時 (split_by_record 時)

```
Val(emb 84 recs): rec_acc=0.4521 macro=0.4312 intra=0.15 inter=0.02 gap=0.13
```

| メトリクス | 意味 |
|---|---|
| `rec_acc` | レコード内 nearest-centroid 精度 (micro 平均)。**主指標** |
| `macro` | レコードごとの精度を均等に平均 (macro 平均) |
| `intra` | 同一話者内のコサイン類似度 (高いほど良い) |
| `inter` | 同レコード内の別話者の重心間コサイン類似度 (低いほど良い) |
| `gap` | intra - inter (大きいほど話者分離が良い) |

評価は**レコード内の 3〜5 人の中から正しい話者を当てる精度**で行い、
最終タスク（1 つの音声データ内の話者識別）に直接対応する。

### 3. 話者時間統計

```bash
python metric_learning/speaker_stats.py
python metric_learning/speaker_stats.py --top 30 --sort segments
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--annotation` | `input/train_annotation.csv` | アノテーション CSV |
| `--top` | `20` | 上位/下位の表示クラス数 |
| `--sort` | `total` | ソート基準: `total` / `segments` / `mean_seg` |

### 4. リファレンス埋め込み辞書の構築

```bash
python metric_learning/build_embedding_dict.py \
  --data_dir ./metric_learning/data \
  --checkpoint ./metric_learning/checkpoints/best_model.pt \
  --output ./metric_learning/checkpoints/speaker_embeddings.npz \
  --aggregation outlier_trimmed
```

| 集約方法 | 動作 |
|---|---|
| `mean` | 全サンプルの単純平均 |
| `median` | 次元ごとの中央値 |
| `trimmed_mean` | コサイン距離上下 fraction を除外して平均 |
| `outlier_trimmed` (推奨) | 下位 fraction を除外して平均。ノイズに強い |

### 5. 推論

```bash
python metric_learning/inference.py \
  --checkpoint ./metric_learning/checkpoints/best_model.pt \
  --audio ./input/train/0HSiCLDz8l/voiceprints/A.wav \
  --load_embeddings ./metric_learning/checkpoints/speaker_embeddings.npz \
  --top_k 5
```

推論時は SubCenter ArcFace ヘッドを使わず、`extract_embedding()` で
192 次元 L2 正規化済みベクトルを出力し、コサイン類似度で話者を検索する。
Multi-crop (center / left / right shift ±200ms) に対応。

## バックグラウンド実行 (PowerShell)

```powershell
$py = ".\venv\Scripts\python.exe"
$trainArgs = 'metric_learning/train.py --data_dir ./metric_learning/data --output_dir ./metric_learning/checkpoints --epochs 50 --batch_size 24 --lr 3e-4 --warmup_epochs 3 --margin_schedule "0,0.05,0.1,0.15,0.2,0.25,0.3,0.35" --scale_schedule "20,32,40,48" --schedule_epochs 24 --max_samples_per_class 100 --balance_classes --balance_samples_per_class 20 --log_file ./metric_learning/checkpoints/train_log.txt --gradient_checkpointing'
Start-Process -NoNewWindow -FilePath $py -ArgumentList $trainArgs -WorkingDirectory (Get-Location) -RedirectStandardOutput ./metric_learning/checkpoints/stdout.txt -RedirectStandardError ./metric_learning/checkpoints/stderr.txt
```

※ `$args` は PowerShell の予約変数のため、引数文字列には `$trainArgs` など別名を使う
※ `--hard_negative` 有効時は ClassBalancedSampler は無効になる
※ `--split_by_record` はデフォルト ON（レコード単位分割）
