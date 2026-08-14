import os, torch, json, fire, glob
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel
from huggingface_hub import snapshot_download

def main(model_repo_name='TIGER-Lab/VideoScore-v1.1', frames_dir='', result_file='res.json', num_chunks=1, chunk_idx=0, **kwargs):
    device = f'cuda'
    print(f'GPU {os.environ.get("CUDA_VISIBLE_DEVICES")} 正在初始化 8B 引擎...')
    
    # 1. 加载身体 (Backbone)
    p = AutoProcessor.from_pretrained(model_repo_name, torch_dtype=torch.bfloat16, use_fast=False)
    m = AutoModel.from_pretrained(model_repo_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    m.eval()
    
    # 2. 地毯式搜索打分权重 (Score Head)
    local_dir = snapshot_download(model_repo_name, local_files_only=True)
    weight_files = glob.glob(f"{local_dir}/*.safetensors") + glob.glob(f"{local_dir}/*.bin")
    score_w, score_b = None, None
    for wf in weight_files:
        if wf.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(wf)
        else:
            sd = torch.load(wf, map_location="cpu")
        if 'score.weight' in sd:
            score_w = sd['score.weight'].to(torch.bfloat16).to(device)
            score_b = sd['score.bias'].to(torch.bfloat16).to(device)
            break

    # 3. 准备 Physics-IQ 任务
    all_f = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    sz = len(all_f) // num_chunks
    st, en = chunk_idx * sz, (len(all_f) if chunk_idx == num_chunks - 1 else (chunk_idx + 1) * sz)
    my_f = all_f[st:en]
    
    # Idefics2 标准对话格式
    prompt = 'User:<image>rate the quality of the video.Assistant:'
    res = []

    for fn in tqdm(my_f, desc=f'GPU {os.environ.get("CUDA_VISIBLE_DEVICES")}'):
        try:
            img = Image.open(os.path.join(frames_dir, fn)).convert('RGB')
            inputs = p(text=[prompt], images=[[img]], return_tensors='pt').to(device, torch.bfloat16)
            with torch.no_grad():
                out = m(**inputs)
                # 核心逻辑：手动执行 Regression Head 运算
                last_hidden = out.last_hidden_state[:, -1, :]
                # VideoScore v1.1 输出 5 个维度：
                # [Visual Quality, Temporal Consistency, Object Interaction, Motion Smoothness, Dynamic Degree]
                raw_scores = (torch.matmul(last_hidden, score_w.T) + score_b).squeeze()
                
                res.append({
                    'video_id': fn.replace('.jpg', ''),
                    'final_score': raw_scores.mean().item(), # 总分取均值
                    'visual_quality': raw_scores[0].item(),
                    'temporal_consistency': raw_scores[1].item(),
                    'object_interaction': raw_scores[2].item(),
                    'motion_smoothness': raw_scores[3].item(),
                    'dynamic_degree': raw_scores[4].item()
                })
        except Exception as e: print(f'Error {fn}: {e}')
        
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f_out: json.dump(res, f_out, indent=4)

if __name__ == "__main__": fire.Fire(main)
