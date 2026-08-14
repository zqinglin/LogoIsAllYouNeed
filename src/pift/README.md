# PIFT: Patch-Invariant Fine-Tuning

Code and evaluation artifacts for the PIFT experiment in the paper (§Cure). PIFT
fine-tunes VideoScore's backbone (Mantis-Idefics2-8B) under two conditions that
are identical except for whether the training patch is correlated with the
video's quality label:

- **C+ (contaminated):** the Sora watermark is applied only to the
  high-VQ clips, exactly reproducing the shortcut we audit in the paper.
- **PIFT (decorrelated):** a diverse patch (Sora / mirrored / gray box / random
  rectangle at a jittered position) is applied to a *VQ-balanced* subset, so
  patch presence carries no information about quality.

Both models share backbone, base clips, labels, and training recipe. Only the
`patch ↔ quality` correlation differs.

## Directory layout

```
pift/
├── README.md                       (this file)
├── environment.lock.yml            conda env spec for PIFT training (name: mantis_train)
├── requirements.lock.txt           pip-only fallback for the same env
├── prep_pift_data.py               step 1: build training / eval JSONs from VideoFeedback
├── launch_pift.sh                  step 2a: 4×A100 full fine-tune (zero3)
├── launch_pift_2gpu.sh             step 2b: 2×A100 fine-tune (zero3 + offload)
├── score_pift.sh                   step 3: score trained models on SAP set (clean vs. watermarked)
├── configs/
│   ├── dataconfig_pift.yaml        Mantis data-config for the PIFT split
│   └── train_pift.json             12,000 VQ-balanced PIFT training samples (from VideoFeedback)
└── results/
    ├── pift_eval_per_video.csv     per-video scores (contaminated vs. PIFT, clean vs. WM)
    ├── acc_full_pift.json          Pearson/Spearman on 2,500 held-out clean test videos
    └── robust_scores/
        ├── pift_clean.csv          PIFT model scores on SAP clean set
        └── pift_wm.csv             PIFT model scores on SAP watermarked set
```

## Environment: use a separate `mantis_train` env (NOT the top-level `videoscore3`)

PIFT fine-tuning has a **different pinned dependency set** from the paper's
evaluation pipeline. The top-level `Code/environment.lock.yml` targets the
`videoscore3` environment used for scoring; do not use it for training here.
Instead create the training environment from the two lock files that live
alongside this README:

```bash
# Recommended (conda; env name is 'mantis_train'):
conda env create -f environment.lock.yml
conda activate mantis_train

# pip-only fallback:
pip install -r requirements.lock.txt
```

Key pinned versions (differences from `videoscore3` in **bold**):

| package     | mantis_train (this env) | videoscore3 (top level) |
|-------------|-------------------------|--------------------------|
| torch       | 2.6.0+cu124             | 2.6.0+cu124              |
| transformers| **4.45.2**              | 4.47.1                   |
| accelerate  | **1.1.0**               | 1.13.0                   |
| datasets    | **4.8.5**               | 2.18.0                   |
| deepspeed   | **0.14.4**              | (not installed)          |
| peft        | **0.12.0**              | (not installed)          |
| numpy       | **1.26.4**              | 2.2.6                    |
| mantis-vl   | 0.0.5 (editable install of the Mantis repo below) | 0.0.5 |

The `mantis-vl` package is an **editable install** of the Mantis training
framework; it is pinned in `requirements.lock.txt` as
`-e git+https://github.com/TIGER-AI-Lab/Mantis.git@f3a3192...#egg=mantis_vl`,
so `pip install -r requirements.lock.txt` will `git clone` it for you at the
exact commit used in the paper. The `launch_pift*.sh` scripts assume this
clone lives at `./Mantis/` (i.e. `src/pift/Mantis/`) — either let the
editable install place it there or clone it manually to that path.

## What is NOT included, and how to obtain it

The following files are excluded because of their size, licensing, or the fact
that they can be regenerated from the artifacts above:

| Item | Approx. size | How to get it |
|---|---|---|
| `pift_images/`, `contam_images/`, `test_images/` (patched training frames) | ~30 GB | Regenerate by running `prep_pift_data.py` (see step 1). Deterministic given the same seed (`--seed 1234`). |
| `ckpt_pift_pift/`, `ckpt_pift_contam/` (trained checkpoints) | ~60 GB total | Retrain via `launch_pift.sh` (~5 h on 4×A100). Public HuggingFace release planned. |
| `Mantis/` (training framework) | ~50 MB | Installed automatically by `pip install -r requirements.lock.txt` (editable install at the paper's exact commit). If cloning manually, place at `src/pift/Mantis/`. |
| `base_model_idefics2/` (backbone) | ~16 GB | `huggingface-cli download TIGER-Lab/Mantis-8B-Idefics2` (auto-cached to `~/.cache/huggingface`). |
| VideoFeedback source dataset | ~200 GB | `huggingface-cli download TIGER-Lab/VideoFeedback` (see the [VideoScore repo](https://github.com/TIGER-AI-Lab/VideoScore) for details). |
| SAP evaluation videos | ~5 GB | Reproduce with `../watermark_tools/` on the paper's 393 base videos; the paper's Appendix B lists the full itemization. |

## Reproducing PIFT end-to-end

### Step 1: Prepare data

Requires VideoFeedback frames (extracted from the HF dataset) at
`$VIDEOFEEDBACK_ROOT/images/` and a copy of the Sora watermark PNG.

```bash
python prep_pift_data.py \
  --train-json  $VIDEOFEEDBACK_ROOT/train_regression.json \
  --images-root $VIDEOFEEDBACK_ROOT \
  --sora-wm     ../watermark_tools/sora-watermark-adder/public/watermarks/sora_watermark.png \
  --out         ./pift_data \
  --n-train     12000 \
  --n-test      2500 \
  --vq-high     3.0 \
  --seed        1234
```

Produces `train_contam.json`, `train_pift.json`, `test_clean.json`, and the
three `*_images/` directories under `./pift_data`.

### Step 2: Fine-tune (both conditions)

Requires the `mantis_train` env created above (which installs Mantis at the
paper's pinned commit) and the Idefics2 backbone cached locally. Both scripts
`source ~/anaconda3/etc/profile.d/conda.sh && conda activate mantis_train` at
their top, so activate the env yourself or edit the source line as needed.

```bash
# 4×A100 (recommended)
bash launch_pift.sh contam 0,1,2,3   # ~2 h
bash launch_pift.sh pift   0,1,2,3   # ~2 h

# 2×A100 fallback (uses zero3 offload)
bash launch_pift_2gpu.sh contam 0,1
bash launch_pift_2gpu.sh pift   0,1
```

Both scripts write into `ckpt_pift_{contam,pift}/`.

### Step 3: Score on SAP

Requires the SAP evaluation set (clean and watermarked video directories) and
the VideoScore repo (for the scoring driver). Point at them via env vars:

```bash
export SAP_CLEAN_DIR=/path/to/sap/clean
export SAP_WM_DIR=/path/to/sap/watermarked
export SAP_PROMPT_CSV=/path/to/video_to_prompt_full.csv
export VIDEOSCORE_ROOT=/path/to/VideoScore
export SCORE_DRIVER=/path/to/score_exp2.py

bash score_pift.sh
```

Writes per-video scores under `exp1_scores/results_{contam,pift}_{clean,wm}.csv`.

## Reading `results/`

- `acc_full_pift.json`: **accuracy** — Pearson/Spearman with human labels on
  the 2,500 held-out clean videos. PIFT matches or exceeds the contaminated
  model, confirming that decontaminating the training data does *not* trade
  accuracy for robustness.
- `pift_eval_per_video.csv`: per-video paired scores used to compute the
  robustness numbers in the paper (Δ Visual Quality between clean and
  watermarked). Both models are evaluated on the same 390 SAP pairs.
- `robust_scores/pift_{clean,wm}.csv`: the raw PIFT scores that go into the
  paired-Δ table (Fig. `fig:pift`).

## Path portability

All paths in the shell scripts default to locations relative to this directory
(`ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd)}"`) and can be overridden via
environment variables. If you keep training data on a separate volume, set
`SAP_CLEAN_DIR`, `SAP_WM_DIR`, `SAP_PROMPT_CSV`, `VIDEOSCORE_ROOT`, and
`SCORE_DRIVER` accordingly.
