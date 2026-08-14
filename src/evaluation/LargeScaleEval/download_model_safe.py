import os
from huggingface_hub import snapshot_download

# 使用强制环境变量让其更稳定
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # 关闭某些复杂的传输协议
os.environ["HF_HUB_DOWNLOAD_MAX_RETRIES"] = "10" # 如果断流自动重连 10 次

MODEL_ID = "TIGER-Lab/Mantis-8B-Idefics2"

print(f"Starting robust download of {MODEL_ID}...")

# max_workers=1 强制它不要并发下载多个文件，老老实实一个接一个下，极大降低被服务器掐断的概率
snapshot_download(
    repo_id=MODEL_ID,
    max_workers=1,
    resume_download=True
)
print("Download finished perfectly!")
