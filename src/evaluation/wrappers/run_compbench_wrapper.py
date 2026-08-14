
import os
import subprocess
import pandas as pd
import re
import sys
from pathlib import Path

# --- [START] User Configuration ---

PYTHON_EXECUTABLE = os.environ.get("PYTHON_EXECUTABLE", sys.executable)

# GPU to use for this evaluation
GPU_ID = "1"

# --- [END] User Configuration ---

# --- Path Definitions ---
CODE_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = os.environ.get("COMPBENCH_BASE_DIR", str(CODE_ROOT / "third_party/T2V-CompBench"))

# The evaluation script to be called
EVAL_SCRIPT = os.path.join(BASE_DIR, "LLaVA/llava/eval/compbench_eval_consistent_attr.py")

# The script to add watermarks
WATERMARK_ADDER_SCRIPT = os.environ.get(
    "WATERMARK_ADDER_SCRIPT",
    str(CODE_ROOT / "src/watermark_tools/add_watermarks.sh"),
)

# --- Data and Output Paths ---
# Source videos that come with the benchmark
ORIGINAL_VIDEO_DIR = os.path.join(BASE_DIR, "video/consistent_attr")

# Directory where watermarked versions will be created
WATERMARKED_VIDEO_DIR = os.path.join(BASE_DIR, "video_watermarked/consistent_attr")

# Path to the prompt metadata file
PROMPT_FILE = os.path.join(BASE_DIR, "meta_data/consistent_attribute_binding.json")

# Centralized directory for all evaluation outputs
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")


def run_command(command, cwd=None):
    """Runs a command and prints its output in real-time."""
    print(f"\n[RUNNING]: {' '.join(command)}")
    print(f"-- In Directory: {cwd or os.getcwd()}\n")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.wait()
    print("\n[FINISHED] ---")
    return process.returncode

def prepare_watermarked_videos():
    """Creates watermarked copies of the original videos."""
    print("--- Step 1: Preparing Watermarked Videos ---")
    
    # Create the target directory
    os.makedirs(WATERMARKED_VIDEO_DIR, exist_ok=True)

    env = os.environ.copy()
    env["INPUT_DIR"] = ORIGINAL_VIDEO_DIR
    env["OUTPUT_DIR"] = WATERMARKED_VIDEO_DIR
    process = subprocess.Popen(
        ["bash", WATERMARK_ADDER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.wait()
    returncode = process.returncode

    if returncode != 0:
        print("\n[ERROR] Watermarking script failed. Aborting.")
        return False
    
    print("\nWatermarked videos created successfully.")
    return True

def run_compbench_evaluation():
    """Runs the T2V-CompBench evaluation for both original and watermarked sets."""
    print("\n--- Step 2: Running T2V-CompBench Evaluation ---")
    
    # Setup the model path command, to be run by the eval script
    model_path_cmd = 'export REAL_PATH=$(find ~/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b -name "config.json" | xargs dirname)'

    datasets_to_evaluate = {
        "original": ORIGINAL_VIDEO_DIR,
        "watermarked": WATERMARKED_VIDEO_DIR
    }

    for name, video_path in datasets_to_evaluate.items():
        print(f"\n--- Evaluating: {name.upper()} Videos ---")
        
        # Define environment variables for the subprocess
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = GPU_ID

        # Construct the command for the evaluation script
        eval_command = [
            PYTHON_EXECUTABLE,
            EVAL_SCRIPT,
            "--model-path", "$REAL_PATH", # The script will resolve this from the env
            "--video-path", video_path,
            "--output-path", OUTPUT_DIR,
            "--read-prompt-file", PROMPT_FILE,
            "--t2v-model", name, # Use 'original' or 'watermarked' for the output filename
            "--temperature", "0.2",
            "--conv-mode", "llava_v1"
        ]
        
        # The final command needs to be run in a shell to handle the variable expansion
        full_command_str = f"{model_path_cmd} && {' '.join(eval_command)}"
        
        # Run the evaluation
        process = subprocess.Popen(full_command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=BASE_DIR)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[ERROR] Evaluation failed for '{name}' set.")
        else:
            print(f"\nEvaluation successful for '{name}' set.")

def analyze_and_summarize_results():
    """Reads the output CSVs and creates a summary report."""
    print("\n--- Step 3: Analyzing and Summarizing Results ---")
    summary_content = """
# T2V-CompBench Evaluation Summary

This report compares the performance on the 'consistent attribute binding' task 
between the original videos and their watermarked versions.

"""
    
    all_scores = {}

    for name in ["original", "watermarked"]:
        csv_path = os.path.join(OUTPUT_DIR, f"{name}_consistent_attr_score.csv")
        try:
            df = pd.read_csv(csv_path)
            # The final score is in the last row, second column
            final_score_str = df.iloc[-1, 0]
            final_score = float(re.search(r"[\d.]+", final_score_str).group())
            
            # The per-video scores are in the last column before the final summary row
            per_video_scores = pd.to_numeric(df['Score'][:-1], errors='coerce').dropna()
            
            mean_score = per_video_scores.mean()
            std_dev = per_video_scores.std()
            
            all_scores[name] = {
                "final_score": final_score,
                "mean_per_video": mean_score,
                "std_dev_per_video": std_dev
            }

            summary_content += f"## Results for: {name.upper()}\n"
            summary_content += f"- Final Official Score: {final_score:.4f}\n"
            summary_content += f"- Average Per-Video Score: {mean_score:.4f}\n"
            summary_content += f"- Standard Deviation: {std_dev:.4f}\n\n"

        except (FileNotFoundError, IndexError, ValueError, TypeError) as e:
            summary_content += f"## Results for: {name.upper()}\n"
            summary_content += f"- Could not process results. File may not exist or is malformed.\n"
            summary_content += f"- Error: {e}\n\n"
    
    # --- Comparison ---
    if "original" in all_scores and "watermarked" in all_scores:
        orig_score = all_scores["original"]["final_score"]
        water_score = all_scores["watermarked"]["final_score"]
        difference = water_score - orig_score
        
        summary_content += "## Comparison Summary\n"
        summary_content += f"- Original Score: {orig_score:.4f}\n"
        summary_content += f"- Watermarked Score: {water_score:.4f}\n"
        summary_content += f"- **Score Difference (Watermarked - Original): {difference:+.4f}**\n"
        if difference > 0:
            summary_content += "- **Conclusion:** Adding a watermark appears to **increase** the evaluation score.\n"
        elif difference < 0:
            summary_content += "- **Conclusion:** Adding a watermark appears to **decrease** the evaluation score.\n"
        else:
            summary_content += "- **Conclusion:** Adding a watermark has **no impact** on the evaluation score.\n"

    summary_path = os.path.join(OUTPUT_DIR, "compbench_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"Analysis complete. Summary report saved to: {summary_path}")

if __name__ == "__main__":
    print("--- T2V-CompBench Evaluation Pipeline Started ---")
    
    # Check if watermarked videos already exist to avoid re-creating them.
    watermarked_videos_exist = False
    if os.path.exists(WATERMARKED_VIDEO_DIR):
        if any(f.endswith('.mp4') for f in os.listdir(WATERMARKED_VIDEO_DIR)):
            watermarked_videos_exist = True

    if watermarked_videos_exist:
        print("\n--- Step 1: Preparing Watermarked Videos ---")
        print(f"Found existing videos in '{WATERMARKED_VIDEO_DIR}'. Skipping creation.")
    else:
        # Step 1: Create watermarked videos if not found
        if not prepare_watermarked_videos():
            import sys
            sys.exit(1)

    # Step 2: Run the evaluations on both sets
    run_compbench_evaluation()
    
    # Step 3: Analyze and compare the results
    analyze_and_summarize_results()
    
    print("\n--- T2V-CompBench Evaluation Pipeline Finished! ---")
