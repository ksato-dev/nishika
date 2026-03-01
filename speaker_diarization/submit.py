"""テスト音声に対する話者ダイアリゼーション推論と提出用CSVの出力。"""

import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import glob
import polars as pl
from tqdm import tqdm

SUBMISSION_COLS = [
    "audio_id",
    "start_time",
    "end_time",
    "target",
]


def main() -> None:
    from utils import SpeakerDiarizer

    base_dir = Path(__file__).resolve().parent

    test_audio_files = sorted(
        glob.glob(os.path.join(base_dir, "input", "test", "**", "*.wav"))
    )

    diarizer = SpeakerDiarizer()

    audio_dirs = list({str(Path(f).parent) for f in test_audio_files})
    diarizer.preload_all_voiceprints(audio_dirs)

    n = len(test_audio_files)
    dfs: list[pl.DataFrame] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        next_prep_future: Future = pool.submit(diarizer.prepare_vad, test_audio_files[0])

        for i in tqdm(range(n), desc="test推論"):
            prepared = next_prep_future.result()
            if i + 1 < n:
                next_prep_future = pool.submit(diarizer.prepare_vad, test_audio_files[i + 1])
            dfs.append(diarizer.diarize_from_prepared(prepared).select(SUBMISSION_COLS))

    submission_df = (
        pl.concat(dfs)
        .sort(by=["audio_id", "start_time"])
        .select(SUBMISSION_COLS)
    )
    out_path = os.path.join(base_dir, "submission_debug.csv")
    submission_df.write_csv(out_path)
    print(f"保存: {out_path}  ({len(submission_df)} 行)")


if __name__ == "__main__":
    main()
