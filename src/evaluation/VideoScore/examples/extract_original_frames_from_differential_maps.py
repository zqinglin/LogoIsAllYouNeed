import argparse
import os
from pathlib import Path
from typing import List, Optional

import av
import pandas as pd
from PIL import Image


WORKSPACE_ROOT = Path(".")
DEFAULT_SUMMARY_CSV = WORKSPACE_ROOT / "evaluation_results/differential_analysis/differential_analysis_summary_all_workers.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "evaluation_results/differential_analysis/original_frames_from_maps"

WM_VIDEO_SEARCH_DIRS = [
    WORKSPACE_ROOT / "Videos/GenVideos/my_videos/alpha_gradients_videos",
    WORKSPACE_ROOT / "Videos/GenVideos/my_videos/watermarked_videos",
    WORKSPACE_ROOT / "Videos/GenVideos/my_videos/gemini_videos",
    WORKSPACE_ROOT / "Videos/GenVideos/my_videos/kling_videos",
    WORKSPACE_ROOT / "Videos/GenVideos/my_videos/gray_videos",
    WORKSPACE_ROOT / "Videos/GenVideos/my_videos/sora_flipped_videos",
]

ORIG_VIDEO_SEARCH_DIRS = [
    WORKSPACE_ROOT / "Videos/GenVideos/my_videos/Videos",
]


def normalize_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (WORKSPACE_ROOT / path).resolve()


def find_summary_row(df: pd.DataFrame, map_path: Path) -> Optional[pd.Series]:
    map_path_norm = map_path.as_posix()
    matches = df[df["output_path"].astype(str).str.replace("\\", "/") == map_path_norm]
    if not matches.empty:
        return matches.iloc[0]

    # Fallback: match by suffix for relative/shifted path inputs.
    suffix = "/" + "/".join(map_path.parts[-4:]) if len(map_path.parts) >= 4 else map_path.name
    matches = df[df["output_path"].astype(str).str.replace("\\", "/").str.endswith(suffix)]
    if not matches.empty:
        return matches.iloc[0]
    return None


def get_video_search_dirs_for_map(map_path: Path) -> List[Path]:
    map_path_s = map_path.as_posix()
    if "/differential_analysis_original/" in map_path_s:
        return ORIG_VIDEO_SEARCH_DIRS + WM_VIDEO_SEARCH_DIRS
    return WM_VIDEO_SEARCH_DIRS + ORIG_VIDEO_SEARCH_DIRS


def resolve_video_path(video_filename: str, search_dirs: List[Path]) -> Optional[Path]:
    for d in search_dirs:
        candidate = d / video_filename
        if candidate.exists():
            return candidate
    return None


def extract_frame(video_path: Path, frame_idx: int) -> Image.Image:
    container = av.open(str(video_path))
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i == frame_idx:
                return Image.fromarray(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    raise RuntimeError(f"Frame {frame_idx} not found in {video_path}")


def process_map_paths(map_paths: List[str], summary_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw in map_paths:
        map_path = normalize_path(raw)
        row = find_summary_row(df, map_path)
        if row is None:
            print(f"[WARN] No summary row found for map: {map_path}")
            continue

        video_filename = str(row["video_filename"])
        frame_idx = int(row["frame_idx"])
        search_dirs = get_video_search_dirs_for_map(map_path)
        video_path = resolve_video_path(video_filename, search_dirs)
        if video_path is None:
            print(f"[WARN] Cannot locate source video '{video_filename}' in configured directories.")
            continue

        image = extract_frame(video_path, frame_idx)
        source_tag = "original" if any(video_path.is_relative_to(d) for d in ORIG_VIDEO_SEARCH_DIRS) else "watermarked"
        out_name = f"{Path(video_filename).stem}_frame_{frame_idx:04d}_{source_tag}.png"
        out_path = output_dir / out_name
        image.save(out_path)
        print(f"[OK] {map_path} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract original video frames corresponding to differential map images.")
    parser.add_argument(
        "map_paths",
        nargs="+",
        help="One or more differential_map_frame_XXXX.png paths (absolute or workspace-relative).",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path to differential_analysis_summary_all_workers.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for extracted original frames.",
    )
    args = parser.parse_args()

    process_map_paths(args.map_paths, args.summary_csv.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
