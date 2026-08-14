#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path


def check_path(label: str, p: Path, must_exist: bool = True) -> bool:
    if must_exist:
        ok = p.exists()
        status = "OK" if ok else "MISSING"
    else:
        ok = True
        status = "OK" if p.exists() else "OPTIONAL_MISSING"
    print(f"[{status}] {label}: {p}")
    return ok


def check_bin(name: str) -> bool:
    path = shutil.which(name)
    ok = path is not None
    status = "OK" if ok else "MISSING"
    print(f"[{status}] binary `{name}`: {path or 'not found'}")
    return ok


def main() -> int:
    code_root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("DATA_ROOT", str(code_root / "data/videos/GenVideos/my_videos")))
    out_root = Path(os.environ.get("OUTPUT_ROOT", str(code_root / "outputs")))

    print("== Reproducibility Preflight Check ==")
    print(f"CODE_ROOT={code_root}")
    print(f"DATA_ROOT={data_root}")
    print(f"OUTPUT_ROOT={out_root}")
    print()

    ok = True
    ok &= check_path("VideoScore code", code_root / "src/evaluation/VideoScore")
    ok &= check_path("LargeScaleEval code", code_root / "src/evaluation/LargeScaleEval")
    ok &= check_path("Watermark assets", code_root / "src/watermark_tools/sora-watermark-adder/public/watermarks")
    ok &= check_path("Prompt CSV", code_root / "data_metadata/video_to_prompt_full.csv")
    ok &= check_path("Input videos (original)", data_root / "Videos")
    ok &= check_path("Input videos (watermarked)", data_root / "watermarked_videos", must_exist=False)
    ok &= check_path("Input videos (alpha gradients)", data_root / "alpha_gradients_videos", must_exist=False)
    print()

    ok &= check_bin("ffmpeg")
    ok &= check_bin("ffprobe")
    print()

    if not out_root.exists():
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[OK] created OUTPUT_ROOT: {out_root}")

    print()
    if ok:
        print("Preflight PASSED.")
        return 0
    print("Preflight FAILED. Please fix missing items above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
