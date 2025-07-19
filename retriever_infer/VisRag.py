from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
import heapq
import os
from tqdm import tqdm
import numpy as np
import json
import re

def weighted_mean_pooling(hidden, attention_mask):  #hidden [batch,vis_token,embedding]
    # import pdb
    # pdb.set_trace()
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps


    
#     return embeddings
@torch.no_grad()
def encode(text_or_image_list, batch_size=32):
    # 根据输入的类型生成 inputs
    if isinstance(text_or_image_list[0], str):
        inputs = {
            "text": text_or_image_list,
            'image': [None] * len(text_or_image_list),
            'tokenizer': tokenizer
        }
    else:
        inputs = {
            "text": [''] * len(text_or_image_list),
            'image': text_or_image_list,
            'tokenizer': tokenizer
        }

    # 计算总的批次数量
    num_samples = len(text_or_image_list)
    num_batches = (num_samples + batch_size - 1) // batch_size  # 向上取整

    all_embeddings = []

    # 使用 tqdm 包装循环以显示进度条
    for batch_idx in tqdm(range(num_batches), desc="Processing batches", total=num_batches, ncols=100, unit="batch"):
        # 获取当前批次的切片
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)

        batch_inputs = {
            "text": inputs["text"][start_idx:end_idx],
            "image": inputs["image"][start_idx:end_idx],
            "tokenizer": tokenizer
        }

        # 将当前批次输入模型
        outputs = model(**batch_inputs)
        attention_mask = outputs.attention_mask
        hidden = outputs.last_hidden_state

        # 使用加权平均池化来获取表示
        reps = weighted_mean_pooling(hidden, attention_mask)
        
        # 归一化嵌入并将结果转换为 numpy 数组
        batch_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        
        # 将当前批次的 embeddings 添加到所有 embeddings 中
        all_embeddings.append(batch_embeddings)

    # 将所有批次的结果合并
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    return final_embeddings

import argparse

parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="VisRAG", help='Path to the model')

parser.add_argument('--output_path', type=str, default="./retrivers_res/VisRAG.json", help='Path to output json')

# Parse the command line arguments
args = parser.parse_args()


json_file_path = "./LongHisDoc.json"


model_name_or_path = args.model_path


out_json_path= args.output_path


image_directory = './LongHisDoc_IMG'  






tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

total_params = sum(p.numel() for p in model.parameters())


print(f"Total number of parameters: {total_params}")
import pdb
pdb.set_trace()
model.eval()


with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)

for root, dirs, files in os.walk(image_directory):
    for folder_name in dirs:
        folder_path = os.path.join(root, folder_name)
        
        if os.path.isdir(folder_path) and folder_name in classified_data:
            
            image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]

           
            samples = classified_data[folder_name]

            
            all_embeddings = []
            
            
            print("Downloading images...")
            passages = []
            

            for img_path in image_paths:
                
                img = Image.open(img_path)
                passages.append(img)
            print("Images downloaded.")
            embeddings_doc = encode(passages)
            

            
            for sample in tqdm(samples, desc=f"Processing samples in {folder_name}", leave=False):
                question = sample['question']  
                INSTRUCTION = "Represent this query for retrieving relevant documents: "
                queries = [INSTRUCTION + question]
                
                embeddings_query = encode(queries)
                scores = (embeddings_query @ embeddings_doc.T)
                score_ls=scores.tolist()
            
                score_ls_1=score_ls[0]
                top_k_indices = [index for index, value in heapq.nlargest(50, enumerate(score_ls_1), key=lambda x: x[1])]
                

                top_k_files = [image_paths[idx] for idx in top_k_indices]

                top_page = [
                    int(re.search(r'page_(\d+)', os.path.basename(path)).group(1)) 
                    for path in top_k_files
                ]

                sample['top_pages']=top_page
               

                with open(out_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(classified_data, json_file, ensure_ascii=False, indent=4)




