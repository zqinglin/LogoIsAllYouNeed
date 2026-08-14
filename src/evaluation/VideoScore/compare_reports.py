
import os
import pandas as pd

# --- [1. 配置区] ---
EVAL_DIR = "outputs"

# 定义要分析的报告及其对应的水印样式名称
# 注意：'sora' 样式对应的是原始的 comparison_report.csv 文件名
REPORT_FILES = {
    "sora": "comparison_report.csv",
    "gemini": "comparison_report_gemini.csv",
    "kling": "comparison_report_kling.csv",
    "gray": "comparison_report_gray.csv",
}

# --- [2. 主逻辑] ---
def main():
    summary_data = []

    print(f"Reading reports from: {EVAL_DIR}\n")

    for style, filename in REPORT_FILES.items():
        file_path = os.path.join(EVAL_DIR, filename)

        if not os.path.exists(file_path):
            print(f"Skipping: Report for style '{style}' not found at {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)

            if df.empty:
                print(f"Skipping: Report for style '{style}' is empty.")
                continue

            # 计算核心指标的均值
            avg_orig_score = df["orig_total"].mean()
            avg_wm_score = df["wm_total"].mean()
            mean_delta = df["delta"].mean()
            num_videos = len(df)

            summary_data.append({
                "Style": style,
                "Num_Videos": num_videos,
                "Avg_Original_Score": avg_orig_score,
                "Avg_Watermarked_Score": avg_wm_score,
                "Mean_Score_Drift": mean_delta,
            })
            
            print(f"Successfully processed report for style '{style}' ({num_videos} videos).")

        except Exception as e:
            print(f"Error processing report for style '{style}': {e}")

    if not summary_data:
        print("\nNo valid reports found to generate a comparison.")
        return

    # 创建汇总的 DataFrame 并打印
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(4) # 格式化，保留4位小数

    print("\n" + "="*80)
    print(" " * 25 + "Video Quality Comparison Report")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    print("\nAnalysis:")
    print(" - Lower 'Mean_Score_Drift' indicates the watermark has less impact on video quality.")
    print(" - A negative drift means the watermark, on average, lowered the perceived quality score.")
    print("="*80)

if __name__ == "__main__":
    main()
