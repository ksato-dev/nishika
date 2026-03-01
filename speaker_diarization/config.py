"""EDA（探索的データ分析）の実行内容をオンオフするフラグ・パラメータ。"""

# ファイル単位の loudness 計算 → ヒストグラム・棒グラフを出力
EDA_LOUDNESS = False

# VAD 区間ごとの loudness を segment_loudness.csv に出力
EDA_SEGMENT_LOUDNESS = False

# voiceprint ごとの loudness を voiceprint_loudness.csv に出力
EDA_VOICEPRINT_LOUDNESS = False

# ---------- スペクトログラム出力用 ----------
# voiceprints フォルダ指定で一括メルスペクトログラム出力を有効にする
SPECTROGRAM_FOLDER = True
# 単一 wav ファイル + 時間指定でスペクトログラム出力を有効にする
SPECTROGRAM_SINGLE = False

SPECTROGRAM_SR = 16000
SPECTROGRAM_N_MELS = 128
SPECTROGRAM_FMAX = 8000.0
SPECTROGRAM_DPI = 150

# メルスペクトログラム用の入力パス（voiceprints フォルダ）。None で無効
MEL_INPUT_PATH = "./input/train/**/voiceprints"
# MEL_INPUT_PATH = "./input/test/4oiuBSaNTo/voiceprints"
# パワースペクトル用の入力パス（単一 wav ファイル）。None で無効
# SPECTRUM_INPUT_PATH = "./input/test/4oiuBSaNTo/voiceprints/A.wav"
# SPECTRUM_INPUT_PATH = "./input/test/4oiuBSaNTo/voiceprints/B.wav"
# SPECTRUM_INPUT_PATH = "./input/test/4oiuBSaNTo/voiceprints/C.wav"
SPECTRUM_INPUT_PATH = "./input/test/4oiuBSaNTo/voiceprints/"
# 出力先ディレクトリ（None の場合は wav の親の spectrograms/）
SPECTROGRAM_OUTPUT_DIR = "./eda/spectrograms/4oiuBSaNTo"

# ---------- パワースペクトル出力用（時間指定） ----------
# 指定時刻のパワースペクトルを出力（横軸: 周波数、縦軸: パワー）
# WINDOW_SEC: FFT 窓幅（秒）、STEP_SEC: スライドのステップ幅（秒）
SPECTRUM_WINDOW_SEC = 0.03
# 全区間スライド出力モード: True なら STEP_SEC ずつずらして全フレームを出力
SPECTRUM_SCAN_ALL = False
SPECTRUM_STEP_SEC = 0.03
# SCAN_ALL=False のときに使う単一時刻
SPECTRUM_CENTER_SEC = 0.45
# FFT サイズ（ゼロパディングで周波数軸を細かくする）
# None = 窓幅のサンプル数のみ（20ms@16kHz→約50Hz刻み）。8192→約2Hz刻み
SPECTRUM_N_FFT = 2000
# 表示する周波数範囲
SPECTRUM_FREQ_MIN = 0.0
SPECTRUM_FREQ_MAX = 1500
# パワースペクトル曲線（周波数 vs パワー）を周波数軸方向に FFT バンドパスする
# 倍音のギザギザ（高域）と全体傾斜（低域）を落として包絡を抽出
SPECTRUM_BANDPASS_ENABLED = False
# 保持する FFT 係数の範囲（割合）。0.0〜1.0
# KEEP_LOW: 下側カット位置（これ未満を 0 に → 全体傾斜を除去）
# KEEP_HIGH: 上側カット位置（これ超を 0 に → ギザギザを除去）
SPECTRUM_BANDPASS_KEEP_LOW = 0.2
SPECTRUM_BANDPASS_KEEP_HIGH = 0.5

# 移動平均のオンオフ
SPECTRUM_MA_ENABLED = True
# 移動平均の窓幅（周波数ビン数）。大きいほど滑らか
SPECTRUM_MA_WINDOW = 4

# スムージング（Savitzky-Golay フィルタ）のオンオフ
SPECTRUM_SMOOTH_ENABLED = False
# SMOOTH_WINDOW: 窓幅（奇数）。大きいほど滑らか
# SMOOTH_POLYORDER: 多項式の次数（窓幅より小さい値）
SPECTRUM_SMOOTH_WINDOW = 13
SPECTRUM_SMOOTH_POLYORDER = 2

# ---------- 単調減衰フィッティング＆残差プロット ----------
# smoothed 曲線を A*exp(-B*f)+C でフィットし、残差を別画像で出力
DECAY_FIT_ENABLED = False
# フィット・残差を取る周波数上限 (Hz)
DECAY_FIT_FREQ_MAX = 4000.0

# ---------- 音声正規化 ----------
# True のとき、VAD・embedding の前に音声をパワー（RMS）正規化する
AUDIO_POWER_NORMALIZE = False

# ---------- 推論パフォーマンス ----------
# embedding 計算のバッチサイズ（GPU メモリに余裕があれば大きくする）
EMBED_BATCH_SIZE = 64 * 8

# ---------- 再検索（voiceprint 強化） ----------
# True のとき、1回目のマッチングで高類似度だったセグメントの embedding を
# voiceprint に加えて話者モデルを強化し、全セグメントを再マッチングする
REMATCH_ENABLED = False
# 1回目で「高類似度」とみなすコサイン類似度の閾値
REMATCH_SIMILARITY_THRESHOLD = 0.3

# ---------- VAD 最小区間フィルタ ----------
# True のとき、VAD 区間長が閾値未満のセグメントを埋め込み計算前に除外する
VAD_MIN_DURATION_FILTER = True
# 除外する区間長の閾値（秒）。これ未満の区間は捨てる
VAD_MIN_DURATION_SEC = 0.05

# ---------- EDA: VAD + embedding クラスタリング ----------
# True のとき、指定音声ファイルに対し VAD → embedding → クラスタリングを実行し CSV 出力
EDA_VAD_CLUSTERING = False
# クラスタリング対象の音声ファイルパス
EDA_VAD_CLUSTERING_INPUT = "./input/train/0HSiCLDz8l/0HSiCLDz8l.wav"
# クラスタリング手法: "agglomerative" / "spectral"
EDA_VAD_CLUSTERING_METHOD = "agglomerative"
# クラスタ数（None の場合は自動推定）
EDA_VAD_CLUSTERING_N_CLUSTERS = None
# 自動推定時のコサイン類似度閾値（agglomerative のみ）
# コサイン類似度 (-1〜1): 大きいほど厳しい（多クラスタ）、小さいほど緩い（少クラスタ）
# 目安: 0.7=厳しい, 0.4=やや厳しい, 0.0=標準, -0.3=緩い
EDA_VAD_CLUSTERING_SIMILARITY_THRESHOLD = 0.3

# ---------- EDA: VAD セグメントの voiceprint 照合 ----------
# True のとき、VAD セグメントの embedding を voiceprint と照合し一致/不一致を CSV 出力
EDA_VAD_VOICEPRINT_MATCH = True
# voiceprint 照合で「一致」とみなすコサイン類似度の閾値
EDA_VAD_VOICEPRINT_MATCH_THRESHOLD = 0.2

# ---------- LPC フォルマント検出 ----------
# LPC 分析を有効にする（パワースペクトルに LPC 包絡線とフォルマントを重ねて表示）
LPC_ENABLED = False
# LPC 次数（目安: sr/1000 の 2 倍前後。16kHz→18〜22）
LPC_ORDER = 20
# プリエンファシス係数（0.95〜0.98 が標準。0 で無効）
LPC_PREEMPHASIS = 0.97
# フォルマント採用条件 — 周波数範囲 (Hz)
LPC_FORMANT_MIN_FREQ = 200.0
LPC_FORMANT_MAX_FREQ = 5000.0
# フォルマント採用条件 — 帯域幅 (Hz)
LPC_BW_MIN = 50.0
LPC_BW_MAX = 400.0
# 表示するフォルマントの最大本数
LPC_MAX_FORMANTS = 4
# 有声判定: フレームの RMS がこの dB 以下なら LPC をスキップ
LPC_MIN_ENERGY_DB = -40.0
