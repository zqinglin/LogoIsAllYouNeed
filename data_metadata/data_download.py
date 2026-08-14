from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "Rapidata/text-2-video-human-preferences"
folder_path = "Videos"

# 获取所有文件列表
all_files = list_repo_files(repo_id, repo_type="dataset")
video_files = [f for f in all_files if f.startswith(f"{folder_path}/") and f.endswith(".mp4")]

# 用字典来去重：键是 (prompt_id, model_name)
download_list = {}

for file_path in video_files:
    filename = file_path.split("/")[-1] # 例如 0000_hunyuan_1724.mp4
    parts = filename.split("_")
    
    if len(parts) >= 2:
        prompt_id = parts[0]
        model_name = parts[1]
        key = (prompt_id, model_name)
        
        # 如果这个 Prompt+模型的组合还没存过，就记录下来
        if key not in download_list:
            download_list[key] = file_path

# 开始批量下载
print(f"筛选完成，准备下载 {len(download_list)} 个视频...")

for key, file_to_download in download_list.items():
    print(f"正在下载: {file_to_download}")
    import time
    import os
    local_dir = "./my_videos"
    local_file_path = os.path.join(local_dir, file_to_download)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file_to_download,
                repo_type="dataset",
                local_dir=local_dir, # 下载到本地的 my_videos 文件夹
                resume_download=True,    # 开启断点续传
                force_download=False     # 不强制重新下载
            )
            break # 下载成功，跳出重试循环
        except Exception as e:
            # 检查是否是 416 错误 (Requested Range Not Satisfiable)
            # 这通常意味着本地文件已损坏或大小与服务器不一致，但 resume_download 试图从错误位置继续
            error_msg = str(e)
            if "416 Client Error" in error_msg or "Requested Range Not Satisfiable" in error_msg:
                print(f"检测到 416 错误，文件可能已损坏。正在删除本地文件并重试: {local_file_path}")
                try:
                    # 尝试删除可能存在的临时文件或不完整文件
                    if os.path.exists(local_file_path):
                        os.remove(local_file_path)
                    # 同时检查并删除可能存在的 .lock 文件或 .incomplete 文件 (huggingface cache)
                    # 这里简单起见，我们直接在下一次尝试中 force_download=True
                except OSError as os_err:
                    print(f"删除文件失败: {os_err}")

                # 强制重新下载
                try:
                     hf_hub_download(
                        repo_id=repo_id,
                        filename=file_to_download,
                        repo_type="dataset",
                        local_dir=local_dir,
                        resume_download=False,
                        force_download=True
                    )
                     break # 强制下载成功
                except Exception as retry_e:
                     print(f"强制重新下载失败: {retry_e}")
            
            print(f"下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2) # 等待几秒后重试
            else:
                print(f"最终下载失败: {file_to_download}")