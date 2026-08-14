import pandas as pd
import json
import os

# 路径锁定：就是你截图中那个 1784 行的大文件
json_path = './eval_results/physics_iq_official/physics_iq_final_merged.json'

def run_stats():
    if not os.path.exists(json_path):
        print(f"❌ 找不到文件: {json_path}，请核对路径！")
        return

    print(f"🔍 正在读取并统计全量数据...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 定义我们要统计的 6 个指标
    metrics = [
        'final_score', 'visual_quality', 'temporal_consistency', 
        'object_interaction', 'motion_smoothness', 'dynamic_degree'
    ]
    
    # 核心统计：计算均值、标准差、最小最大值
    report = df[metrics].agg(['mean', 'std', 'min', 'max']).T
    
    print("\n" + "="*60)
    print("🚀 Physics-IQ 评测全维度统计报表")
    print("="*60)
    print(f"📊 总样本数: {len(df)}")
    print("-" * 60)
    print(report.to_string())
    print("="*60)

    # 生成 Excel 方便发给学长
    output_excel = "Physics_IQ_Statistics.xlsx"
    with pd.ExcelWriter(output_excel) as writer:
        df.to_excel(writer, sheet_name='RawData', index=False)
        report.to_excel(writer, sheet_name='Summary')
    
    print(f"✅ 大功告成！统计文件已生成: {output_excel}")

if __name__ == "__main__":
    run_stats()