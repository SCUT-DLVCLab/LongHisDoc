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
parser.add_argument('--model_path', type=str, default="SigLip", help='Path to the model')

parser.add_argument('--output_path', type=str, default="./retrivers_res/SigLip.json", help='Path to output json')

# Parse the command line arguments
args = parser.parse_args()


json_file_path = "./LongHisDoc.json"


model_name = args.model_path


out_json_path= args.output_path


image_directory = './LongHisDoc_IMG'  



# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)

# 获取字典的keys并去掉.pdf后缀
# doc_ids = [key.replace('.pdf', '') for key in classified_data.keys()]

# 加载模型
# model_name = "/home/szy/local/DL_Work_space/LLM_code/models/colqwen2-v0.1-merged"

# model_name = "/home/szy/local/DL_Work_space/colpali_model_infer_test/save_models/colqwen2-merege-v1.0-dpan-5e-5-warmup30_100k_guji_2_2/checkpoint-1000"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
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

            # for img in images:
            #     image_processed = processor([img]).to(model.device) #这个processor函数就是会把数据放到cpu上处理，所以一次处理太多图片还是会爆内存
            #     with torch.no_grad():
            #         image_embeddings = model(**image_processed)
            #         # if image_embeddings.shape[1]==759:
            #             # import pdb
            #             # pdb.set_trace()
            #             # print(batch_image_paths)

            #     all_embeddings.append(image_embeddings)

            # # import pdb
            # # pdb.set_trace()    
            # all_embeddings = pade_array(all_embeddings, pad_value=0)
            # all_embeddings = torch.cat(all_embeddings).to(device="cuda:0", dtype=torch.bfloat16)
            # # import pdb
            # pdb.set_trace() 
        


            ###########query处理#############
            # 逐个处理样本
            # if len(images) > 1:
            for sample in tqdm(samples, desc=f"Processing samples in {folder_name}", leave=False):
                # if sample["answer_page_idx"]:
                question = sample['question']  # 读取问题
                inputs = processor(text=[question], images=images, return_tensors="pt", padding=True,truncation=True).to(device)


                # 前向传播
                with torch.no_grad():
                    scores = model(**inputs)
                
                scores = scores.logits_per_image

                # 评分
                # scores = processor.score_multi_vector(query_embeddings, all_embeddings)

                # 找到前20个最大得分的索引
                # 获取 scores 的长度
                # import pdb
                # pdb.set_trace()
                num_scores = scores.size(0)

                # 动态设置 k 值，如果 scores 的长度小于 20，则使用 scores 的长度作为 k
                k = min(50, num_scores)

                # 获取 top_k_scores 和 top_k_indices
                top_k_scores, top_k_indices = torch.topk(scores, k=k, dim=0)
                # import pdb
                # pdb.set_trace()
                # top_k_scores, top_k_indices = torch.topk(scores, k=20)

                # 将前五个最大得分和对应的图片路径添加到样本中
                # sample['top_scores'] = top_k_scores[0].tolist()  # 转换为列表
                top_k_indices = top_k_indices.flatten().tolist()
                top_k_files = [image_paths[idx] for idx in top_k_indices]

                top_page = [
                    int(os.path.basename(path).replace(".jpg","").split("_")[-1]) 
                    for path in top_k_files
                ]
                sample['top_pages']=top_page
                with open(out_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(classified_data, json_file, ensure_ascii=False, indent=4)
               

                

# 将更新后的数据写回JSON文件（如果需要）
# with open(out_json_path, 'w', encoding='utf-8') as json_file:
#     json.dump(classified_data, json_file, ensure_ascii=False, indent=4)

# 输出处理后的doc_ids
# print(doc_ids)

# import json
# import argparse
# import os



# input_json_file='/home/szy/local/DL_Work_space/colpali_model_infer_test/guji_11_25_2_1_test_data_res/colqwen2-merege-v1.0-dpan-5e-4-warmup30_100k_guji_2_2_data_bs_1_1000_steps.json'
