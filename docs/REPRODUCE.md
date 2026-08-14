# Reproduce (Reviewer-Oriented)

This code package is path-portable inside `Code/` and does not require the original absolute paths.

## 1) Prepare Environment

Recommended (matches paper appendix versions):

```bash
conda env create -f Code/environment.lock.yml
conda activate videoscore3
```

Fallback (pip-only):

```bash
pip install -r Code/requirements.lock.txt
```

Required CLI tools:
- `ffmpeg`
- `ffprobe`

## 2) Place Data

By default, scripts expect:
- `Code/data/videos/GenVideos/my_videos/Videos`
- `Code/data/videos/GenVideos/my_videos/watermarked_videos` (generated if absent)
- `Code/data/videos/GenVideos/my_videos/alpha_gradients_videos` (generated if absent)

Prompt mapping:
- `Code/data_metadata/video_to_prompt_full.csv`

If you use another location, set:
- `DATA_ROOT=./relative/path/GenVideos/my_videos`

Example:

```bash
export DATA_ROOT=./Code/data/videos/GenVideos/my_videos
```

## 3) Preflight Check

```bash
python Code/scripts/repro_check.py
```

## 4) Quick Reviewer Demo (Single Pair)

This package includes one original/watermarked video pair for fast validation:

```bash
bash Code/examples/reviewer_demo/run_demo.sh
```

Result JSON:
- `Code/examples/reviewer_demo/output/demo_result.json`

## 5) Generate Watermarked / Alpha Variants

```bash
# Dynamic mask generation
python Code/src/watermark_tools/make_dynamic_mask.py

# Watermarked set
bash Code/src/watermark_tools/add_watermarks.sh

# Alpha sweep set
bash Code/data_metadata/generate_alpha_gradients.sh
```

## 6) Run Core Evaluation

Example (VideoScore v1.1 large-scale):

```bash
python Code/src/evaluation/LargeScaleEval/run_large_videoscore.py
```

Outputs default to:
- `Code/outputs/LargeScaleEval/`

For v1 / official-v1 / baseline VLMs, run corresponding scripts in:
- `Code/src/evaluation/LargeScaleEval/`

## 7) PIFT: Reproduce the Cure Experiment

Full end-to-end recipe (data prep → fine-tune both C+ and PIFT models → SAP
scoring) is documented in `Code/src/pift/README.md`. In short:

```bash
# a. Build the two training splits from VideoFeedback
python Code/src/pift/prep_pift_data.py \
    --train-json  $VIDEOFEEDBACK_ROOT/train_regression.json \
    --images-root $VIDEOFEEDBACK_ROOT \
    --sora-wm     Code/src/watermark_tools/sora-watermark-adder/public/watermarks/sora_watermark.png \
    --out         ./pift_data

# b. Fine-tune both conditions on Mantis-Idefics2-8B (needs a Mantis/ clone)
bash Code/src/pift/launch_pift.sh contam 0,1,2,3
bash Code/src/pift/launch_pift.sh pift   0,1,2,3

# c. Score both on SAP clean vs. watermarked
export SAP_CLEAN_DIR=/path/to/sap/clean SAP_WM_DIR=/path/to/sap/watermarked
bash Code/src/pift/score_pift.sh
```

Precomputed metrics (Pearson/Spearman on the 2,500 held-out clean videos, and
per-video paired scores on the 390 SAP pairs) are already checked in under
`Code/src/pift/results/`, so reviewers can inspect the numbers behind Fig.
`fig:pift` without rerunning the full training pipeline.

## 8) Optional Path Overrides

All key scripts support environment-variable overrides:
- `DATA_ROOT`
- `PROMPT_CSV` or `PROMPT_CSV_PATH`
- `OUTPUT_CSV` or `OUTPUT_DIR`
- `VIDEOSCORE_PROJECT_PATH`
- `VIDEOSCORE2_WORKDIR` / `VIDEOSCORE2_SCRIPT` (for `run_evaluation_wrapper.py`)
- `COMPBENCH_BASE_DIR` (for `run_compbench_wrapper.py`)
- `VIDEOFEEDBACK_ROOT`, `SAP_CLEAN_DIR`, `SAP_WM_DIR`, `SAP_PROMPT_CSV`,
  `VIDEOSCORE_ROOT`, `SCORE_DRIVER` (for `src/pift/`)

This allows running on any machine without editing source files.
