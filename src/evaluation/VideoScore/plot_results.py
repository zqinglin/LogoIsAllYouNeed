import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import matplotlib.patches as mpatches

# --- [START] User Configuration ---
RESULTS_DIR = "outputs"
PLOTS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")
# --- [END] User Configuration ---

def load_and_prepare_data():
    """Loads and prepares the dataframe from CSV files."""
    print("--- Loading and Preparing Data ---")
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*scores_*.csv"))
    if not csv_files:
        print(f"FATAL: No CSV score files found in '{RESULTS_DIR}'. Please run the evaluation first.")
        return None
    print(f"Found {len(csv_files)} score files to process.")
    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['type'] = 'original' if "_original" in f else 'watermarked'
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not read or process file {f}. Error: {e}")
    if not all_data:
        print("FATAL: No valid data could be loaded from CSV files.")
        return None
    return pd.concat(all_data, ignore_index=True)

def create_truly_overlaid_violin_plot(df):
    """Generates a single violin plot with 5 metrics truly overlaid using alpha transparency."""
    print("\n--- Generating Truly Overlaid Violin Plot ---")
    
    score_metrics = [
        'vs1_visual_quality', 
        'vs1_temporal_consistency', 
        'vs1_dynamic_degree', 
        'vs1_text_alignment', 
        'vs1_factual_consistency'
    ]
    
    macaron_palette = sns.color_palette("pastel", n_colors=5)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot each metric as a separate, transparent layer on the same axes
    for i, metric in enumerate(score_metrics):
        sns.violinplot(data=df, x='type', y=metric, ax=ax, color=macaron_palette[i], 
                       inner=None, # No inner plot to keep it clean
                       linewidth=0.8)
    
    # Adjust the alpha for all violins to make them transparent
    for collection in ax.collections:
        collection.set_alpha(0.5)

    ax.set_title('Overlaid Comparison of All Metrics for Original vs. Watermarked Videos',
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Video Type', fontsize=14, labelpad=15)
    ax.set_ylabel('Score Distribution', fontsize=14, labelpad=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=10)

    # Create a custom legend
    legend_patches = []
    for i, metric in enumerate(score_metrics):
        label = metric.replace('vs1_', '').replace('_', ' ').title()
        patch = mpatches.Patch(color=macaron_palette[i], label=label, alpha=0.6)
        legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, title="Metrics", 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_filename = "truly_overlaid_metrics_violin_plot.png"
    output_path = os.path.join(PLOTS_OUTPUT_DIR, output_filename)
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Successfully saved overlaid plot: {output_path}")
    except Exception as e:
        print(f"FATAL: Could not save the plot {output_path}. Error: {e}")
    plt.close(fig)

def main():
    """Main function to run the plotting script."""
    if not os.path.exists(PLOTS_OUTPUT_DIR):
        os.makedirs(PLOTS_OUTPUT_DIR)
        print(f"Created directory: {PLOTS_OUTPUT_DIR}")

    combined_df = load_and_prepare_data()
    if combined_df is not None:
        create_truly_overlaid_violin_plot(combined_df)
        print("\n--- Plot generation complete! ---")

if __name__ == "__main__":
    main()
