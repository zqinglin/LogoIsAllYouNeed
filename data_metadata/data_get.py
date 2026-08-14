import pandas as pd
from datasets import load_dataset

# 1. 加载数据
dataset = load_dataset("Rapidata/text-2-video-human-preferences", split="train")
df = dataset.to_pandas()

# 2. 我们建立一个“全量库”，把 video1 和 video2 都放进去
# 这样无论 ID 在哪一列，或者哪一列漏了 ID，我们都能抓到
v1_map = df[['video1', 'prompt']].rename(columns={'video1': 'video_path'})
v2_map = df[['video2', 'prompt']].rename(columns={'video2': 'video_path'})

# 合并
full_list = pd.concat([v1_map, v2_map])

# 3. 重点：从路径里提取最后的文件名 (比如从 url 里提取 0000_sora_0.mp4)
full_list['filename'] = full_list['video_path'].apply(lambda x: x.split('/')[-1].replace('.gif', '.mp4'))

# 4. 按“文件名”去重，而不是按 ID 去重
# 这样 0000_sora 和 0000_hunyuan 都会被保留
final_mapping = full_list[['filename', 'prompt']].drop_duplicates(subset=['filename'])

# 5. 排序并保存
final_mapping = final_mapping.sort_values('filename')
final_mapping.to_csv("video_to_prompt_full.csv", index=False)

print(f"处理完成！现在你的 CSV 里应该有 {len(final_mapping)} 条记录了，涵盖了所有不同的视频。")