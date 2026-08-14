import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from pathlib import Path

# --- [1. 配置区] ---
CODE_ROOT = Path(__file__).resolve().parents[4]
INPUT_CSV = os.environ.get(
    "INPUT_CSV",
    str(CODE_ROOT / "outputs/LargeScaleEval/results_videoscore_with_variance.csv"),
)
PLOTS_OUTPUT_DIR = os.environ.get(
    "PLOTS_OUTPUT_DIR",
    str(CODE_ROOT / "outputs/LargeScaleEval/final_comparison"),
)
OUTPUT_FILENAME = "original_vs_watermark_violin.pdf"

# --- [2. 核心函数] ---

def transform_data_for_plotting(df):
    print("--- Sorting Original vs. Watermarked data ---")
    
    # 🎯 终极防弹分类逻辑
    def get_video_type(filename):
        filename = str(filename)
        # 用 endswith 严格匹配结尾，防止被模型名或文件夹路径里的 alpha 误伤
        if filename.endswith('sora_watermark.mp4'):
            return 'watermarked'
        elif re.search(r'_alpha_[0-9]\.[0-9]\.mp4$', filename):
            return None # 精准排除掉所有的攻击梯度组
        elif filename.endswith('.mp4'):
            return 'original' # 剩下的纯净 .mp4 绝对是原始视频！
        return None

    df['type'] = df['video'].apply(get_video_type)
    filtered_df = df[df['type'].notnull()].copy()
    
    # 打印一下提取结果，让你心里有数
    orig_count = len(filtered_df[filtered_df['type']=='original'])
    wm_count = len(filtered_df[filtered_df['type']=='watermarked'])
    print(f"📊 成功捕捉: {orig_count} 个原始视频, {wm_count} 个水印视频")

    metric_cols = {
        'visual_quality_mean': 'Visual Quality',
        'temporal_consistency_mean': 'Temporal Consistency',
        'dynamic_degree_mean': 'Dynamic Degree',
        'text_to_video_alignment_mean': 'Text Alignment',
        'factual_consistency_mean': 'Factual Consistency'
    }

    df_selected = filtered_df[['type'] + list(metric_cols.keys())].copy()
    df_selected.rename(columns=metric_cols, inplace=True)
    melted_df = df_selected.melt(id_vars=['type'], var_name='metric', value_name='score')
    
    return melted_df

def create_overlaid_plot(df):
    if df is None or df.empty: return
    
    print("--- Generating academic Overlaid PDF Vector plot ---")
    
    metrics = df['metric'].unique()
    macaron_palette = sns.color_palette("pastel", n_colors=len(metrics))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # 严格 1:1 还原学长图一的重叠画法
    for i, metric in enumerate(metrics):
        metric_df = df[df['metric'] == metric]
        sns.violinplot(data=metric_df, x='type', y='score', ax=ax, 
                       color=macaron_palette[i], inner=None, linewidth=0.8,
                       order=['original', 'watermarked'])

    # 学长的灵魂半透明度
    for collection in ax.collections:
        collection.set_alpha(0.5)

    # 底部/顶部无标题，干干净净留给 LaTeX
    ax.set_xlabel('Video Category', fontsize=15, labelpad=15)
    ax.set_ylabel('Mean Score', fontsize=15, labelpad=15)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=12)

    # 图例配置
    legend_patches = [mpatches.Patch(color=macaron_palette[i], label=metric, alpha=0.6) for i, metric in enumerate(metrics)]
    ax.legend(handles=legend_patches, title="Evaluation Metrics", title_fontsize='14',
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='12')

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, bbox_inches='tight', format='pdf')
    print(f"\n✅ 完美 LaTeX 矢量图已保存至: {output_path}")
    plt.close(fig)

def main():
    print(f"Reading experimental data from {INPUT_CSV}...")
    try:
        report_df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"FATAL: Error reading CSV: {e}")
        return

    plotting_df = transform_data_for_plotting(report_df)
    create_overlaid_plot(plotting_df)
    print("--- Script finished. ---")

if __name__ == "__main__":
    main()
