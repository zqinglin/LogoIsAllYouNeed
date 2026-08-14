import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- [1. 配置区] ---
RESULTS_DIR = "outputs"
ORIGINAL_CSV = os.path.join(RESULTS_DIR, "videoscore2_scores_original.csv")
WATERMARKED_CSV = os.path.join(RESULTS_DIR, "videoscore2_scores_watermarked.csv")

PLOTS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")
OUTPUT_FILENAME = "videoscore2_overlaid_violin_plot.png"

# --- [2. 核心函数] ---

def load_and_transform_data():
    """加载并转换VideoScore2的数据以进行绘图。"""
    print("--- Loading and transforming VideoScore2 data ---")
    
    try:
        df_orig = pd.read_csv(ORIGINAL_CSV)
        df_wm = pd.read_csv(WATERMARKED_CSV)
    except FileNotFoundError as e:
        print(f"FATAL: Could not find input CSV file. {e}")
        return None

    # 添加type列
    df_orig['type'] = 'original'
    df_wm['type'] = 'watermarked'

    # 合并数据
    combined_df = pd.concat([df_orig, df_wm], ignore_index=True)

    # 定义要绘制的指标列（只使用均值）
    metric_cols = {
        'vs2_visual_quality_mean': 'Visual Quality',
        'vs2_text_alignment_mean': 'Text Alignment',
        'vs2_physical_consistency_mean': 'Physical Consistency'
    }
    
    # 筛选出需要的列
    cols_to_keep = ['video_filename', 'type'] + list(metric_cols.keys())
    plotting_df = combined_df[cols_to_keep].copy()
    
    # 重命名列以获得更清晰的图例
    plotting_df.rename(columns=metric_cols, inplace=True)
    
    # 使用melt将宽数据转换为长数据
    melted_df = plotting_df.melt(
        id_vars=['video_filename', 'type'],
        value_vars=list(metric_cols.values()),
        var_name='metric',
        value_name='score'
    )
    
    print("Data transformation complete.")
    return melted_df

def create_overlaid_plot(df):
    """根据转换后的长格式数据，生成重叠小提琴图。"""
    if df is None:
        return
        
    print("--- Generating VideoScore2 overlaid violin plot ---")
    
    metrics = df['metric'].unique()
    macaron_palette = sns.color_palette("pastel", n_colors=len(metrics))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))

    # 在同一个坐标系上为每个指标画一个半透明的小提琴图层
    for i, metric in enumerate(metrics):
        metric_df = df[df['metric'] == metric]
        sns.violinplot(data=metric_df, x='type', y='score', ax=ax, color=macaron_palette[i], 
                       inner=None, linewidth=0.8)

    for collection in ax.collections:
        collection.set_alpha(0.5)

    ax.set_title('Overlaid Comparison of VideoScore2 Metrics for Original vs. Watermarked Videos',
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Video Type', fontsize=14, labelpad=15)
    ax.set_ylabel('Score Distribution (Mean)', fontsize=14, labelpad=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=10)

    legend_patches = [mpatches.Patch(color=macaron_palette[i], label=metric, alpha=0.6) for i, metric in enumerate(metrics)]
    ax.legend(handles=legend_patches, title="Metrics", 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if not os.path.exists(PLOTS_OUTPUT_DIR):
        os.makedirs(PLOTS_OUTPUT_DIR)
        print(f"Created directory: {PLOTS_OUTPUT_DIR}")
        
    output_path = os.path.join(PLOTS_OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot successfully saved to: {output_path}")
    except Exception as e:
        print(f"FATAL: Could not save the plot. Error: {e}")
    plt.close(fig)

# --- [3. 主程序入口] ---

def main():
    """主函数，执行数据加载、转换和绘图。"""
    plotting_df = load_and_transform_data()
    if plotting_df is not None:
        create_overlaid_plot(plotting_df)
        print("--- Script finished. ---")

if __name__ == "__main__":
    main()
