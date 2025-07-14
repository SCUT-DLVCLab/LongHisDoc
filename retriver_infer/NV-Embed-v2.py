import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}

query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
# queries = [
#     'are judo throws allowed in wrestling?', 
#     'how to become a radiology technician in michigan?'
#     ]

# No instruction needed for retrieval passages
passage_prefix = ""

import argparse

parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="NV-Embed-v2", help='Path to the model')

parser.add_argument('--output_path', type=str, default="./retrivers_res/NV-Embed-v2.json", help='Path to output json')

# Parse the command line arguments
args = parser.parse_args()

input_json_file = "./LongHisDoc.json"

output_json_file = args.output_path

ocr_dir = "./OCR_Res/"



# load model with tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to(device)

# get the embeddings
max_length = 32768



from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
import os
import re
import json

# 向量数据库chroma相关






import torch
import torch.nn.functional as F
from tqdm import tqdm
# from tqdm import tqdm



def find_top_k_similar(texts, query,index_ls, k=25, batch_size=8):
    # 处理文本列表，获取所有文本的 [CLS] 向量
    all_text_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, instruction=passage_prefix, max_length=max_length)
        
        # embeddings = encoder(batch)
        all_text_vecs.append(embeddings)
    

    # 将所有批次的嵌入拼接在一起
    all_text_vecs = torch.cat(all_text_vecs, dim=0)  # Shape: [num_texts, hidden_size]
    all_text_vecs = F.normalize(all_text_vecs, p=2, dim=1)

    # 对查询文本进行嵌入
    query_vec = model.encode(query, instruction=query_prefix, max_length=max_length)  # Shape: [1, hidden_size]

    query_vec = F.normalize(query_vec, p=2, dim=1)

    # 计算余弦相似度
    # 使用 F.cosine_similarity 来计算文本向量和查询向量的相似度
    similarities = (query_vec @ all_text_vecs.T) * 100  # Shape: [num_texts]

    # 获取相似度最高的前 k 个文本的索引
    if k >len(texts):
        k = len(texts)
    top_k_indices = torch.topk(similarities[0], k=k).indices

    # 获取对应的最相似文本
    top_k_texts = [texts[i] for i in top_k_indices]

    top_k_index = [index_ls[i] for i in top_k_indices] 

    return top_k_texts, top_k_index 




with open(input_json_file, 'r') as f:
    data = json.load(f)

for d_key, d_value in tqdm(data.items()):
    

    json_p = ocr_dir + str(d_key) + ".json"
    anno_dic = json.load(open(json_p))
   
    texts = []
    index_ls = []
    for key, value in anno_dic.items():
        
        try:
            texts.append(value["punc"])
        except:
            texts.append("")
        index_ls.append(int(key) + 1)
    
    



    for d in tqdm(d_value):
        
        user_query = d["question"]

        top_k_texts, top_k_index = find_top_k_similar(texts, [user_query],index_ls, k=50, batch_size=2)

        d["top_pages"] = top_k_index 
        

        with open(output_json_file, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
