import os
import json
import torch
from PIL import Image
# from colpali_engine.models import ColQwen2, ColQwen2Processor
from tqdm import tqdm
import re
import numpy as np

from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel

def pade_array(arrays, pad_value=0):
    # 找到最长的第二维长度
    max_second_dim = max(array.shape[1] for array in arrays)
    # third_dim = arrays[0].shape[2]  # 假设第三维大小固定
    
    # 对每个数组进行补长并存入列表
    padded_arrays = [
        np.pad(array.cpu().to(torch.float32).numpy(), 
               ((0, 0), (max_second_dim - array.shape[1], 0), (0, 0)), 
               mode='constant', 
               constant_values=pad_value)
        for array in arrays
    ]
    all_embeddings = [torch.tensor(array) for array in padded_arrays]
    return all_embeddings

# 假设你的JSON文件路径和图片目录路径
import argparse

parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="clip", help='Path to the model')

parser.add_argument('--output_path', type=str, default="./retrivers_res/clip.json", help='Path to output json')

# Parse the command line arguments
args = parser.parse_args()


json_file_path = "./LongHisDoc.json"

model_name = args.model_path

out_json_path= args.output_path



image_directory = './LongHisDoc_IMG'  # 你的图片文件夹路径

batch_size = 1  # 设置批量大小

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)

# 获取字典的keys并去掉.pdf后缀
# doc_ids = [key.replace('.pdf', '') for key in classified_data.keys()]

# 加载模型
# model_name = "/home/szy/local/DL_Work_space/LLM_code/models/colqwen2-v0.1-merged"

# model_name = "/home/szy/local/DL_Work_space/colpali_model_infer_test/save_models/colqwen2-merege-v1.0-dpan-5e-5-warmup30_100k_guji_2_2/checkpoint-1000"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(model_name)

processor = CLIPProcessor.from_pretrained(model_name)
model.to(device)



# 遍历每个文件夹
for root, dirs, files in os.walk(image_directory):
    for folder_name in tqdm(dirs, desc="Processing folders"):
        folder_path = os.path.join(root, folder_name)
        tqdm.write(f"Processing folders: {folder_name}")
        # 只处理存在于doc_ids中的文件夹
        if os.path.isdir(folder_path) and folder_name in classified_data:
            # 获取图片路径
            image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]
            # print(image_paths)
            # 从JSON中获取样本列表
            samples = classified_data[folder_name]
            # if not samples["answer_page_idx"]:
            #     sample['top_pages']=[]
            #     with open(out_json_path, 'w', encoding='utf-8') as json_file:
            #             json.dump(classified_data, json_file, ensure_ascii=False, indent=4)
                
            # else:
                # 批量处理所有图片
            images = [Image.open(image_path) for image_path in image_paths]

            all_embeddings = []

            
        


           
            for sample in tqdm(samples, desc=f"Processing samples in {folder_name}", leave=False):
                # if sample["answer_page_idx"]:
                question = sample['question']  # 读取问题
                inputs = processor(text=[question], images=images, return_tensors="pt", padding=True,truncation=True).to(device)


                # 前向传播
                with torch.no_grad():
                    scores = model(**inputs)
                
                scores = scores.logits_per_image

               
                num_scores = scores.size(0)

                # 动态设置 k 值，如果 scores 的长度小于 20，则使用 scores 的长度作为 k
                k = min(50, num_scores)

                # 获取 top_k_scores 和 top_k_indices
                top_k_scores, top_k_indices = torch.topk(scores, k=k, dim=0)
                
                top_k_indices = top_k_indices.flatten().tolist()
                top_k_files = [image_paths[idx] for idx in top_k_indices]

                top_page = [
                    int(os.path.basename(path).replace(".jpg","").split("_")[-1]) 
                    for path in top_k_files
                ]
                sample['top_pages']=top_page
                with open(out_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(classified_data, json_file, ensure_ascii=False, indent=4)
               

                


