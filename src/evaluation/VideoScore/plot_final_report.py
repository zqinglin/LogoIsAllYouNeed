import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- [1. 配置区] ---
# 输入的CSV报告文件
INPUT_CSV = "outputs/comparison_report.csv"
# 输出图表的目录
PLOTS_OUTPUT_DIR = "outputs/plots"
# 输出图表的文件名
OUTPUT_FILENAME = "final_overlaid_violin_plot.png"

# --- [2. 核心函数] ---

def transform_data_for_plotting(df):
    """将宽格式的报告数据转换为适用于seaborn绘图的长格式数据。"""
    print("--- Transforming data for plotting ---")
    
    # 定义与CSV文件完全匹配的指标映射
    orig_metric_map = {
        'orig_visual': 'Visual Quality',
        'orig_align': 'Text Alignment'
    }
    wm_metric_map = {
        'wm_visual': 'Visual Quality',
        'wm_align': 'Text Alignment'
    }
    
    # 1. 处理原始视频数据
    df_orig = df[['video_filename'] + list(orig_metric_map.keys())].copy()
    df_orig.rename(columns=orig_metric_map, inplace=True)
    df_orig_melted = df_orig.melt(id_vars=['video_filename'], var_name='metric', value_name='score')
    df_orig_melted['type'] = 'original'

    # 2. 处理加水印视频数据
    df_wm = df[['video_filename'] + list(wm_metric_map.keys())].copy()
    df_wm.rename(columns=wm_metric_map, inplace=True)
    df_wm_melted = df_wm.melt(id_vars=['video_filename'], var_name='metric', value_name='score')
    df_wm_melted['type'] = 'watermarked'
    
    # 3. 合并两个长格式数据框
    plotting_df = pd.concat([df_orig_melted, df_wm_melted], ignore_index=True)
    
    print("Data transformation complete.")
    return plotting_df

def create_overlaid_plot(df):
    """根据转换后的长格式数据，生成最终的重叠小提琴图。"""
    print("--- Generating final overlaid violin plot ---")
    
    # 获取指标列表和颜色
    metrics = df['metric'].unique()
    macaron_palette = sns.color_palette("pastel", n_colors=len(metrics))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))

    # 在同一个坐标系上，为每个指标画一个半透明的小提琴图层
    for i, metric in enumerate(metrics):
        metric_df = df[df['metric'] == metric]
        sns.violinplot(data=metric_df, x='type', y='score', ax=ax, color=macaron_palette[i], 
                       inner=None, linewidth=0.8)

    # 调整所有图层的透明度
    for collection in ax.collections:
        collection.set_alpha(0.5)

    # 设置图表标题和标签
    ax.set_title('Overlaid Comparison of All Metrics for Original vs. Watermarked Videos',
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Video Type', fontsize=14, labelpad=15)
    ax.set_ylabel('Score Distribution', fontsize=14, labelpad=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=10)

    # 创建自定义图例
    legend_patches = [mpatches.Patch(color=macaron_palette[i], label=metric, alpha=0.6) for i, metric in enumerate(metrics)]
    ax.legend(handles=legend_patches, title="Metrics", 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # 保存图表
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
    print(f"Reading data from {INPUT_CSV}...")
    try:
        report_df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"FATAL: Input file not found at {INPUT_CSV}. Please run the evaluation script first.")
        return

    if report_df.empty:
        print("Warning: The report file is empty. No plot will be generated.")
        return

    plotting_df = transform_data_for_plotting(report_df)
    create_overlaid_plot(plotting_df)
    print("--- Script finished. ---")

if __name__ == "__main__":
    main()
