# 功能说明：实现使用langchain框架，使用LCEL构建一个完整的LLM应用程序用于RAG知识库的查询，并使用fastapi进行发布
# 包含：langchain框架的使用，langsmith跟踪检测

# 相关依赖库
# pip install langchain langchain-openai langchain-chroma
# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
import os
import re
import json
import asyncio
import uuid
import time
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def read_and_concat_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 去除每行末尾的换行符并用空格拼接
            concatenated_string = ' '.join(line.strip() for line in lines)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
    return concatenated_string



# 使用示例


import argparse

parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="all-MiniLM-L6-v2", help='Path to the model')

parser.add_argument('--output_path', type=str, default="./retrivers_res/all-MiniLM-L6-v2.json", help='Path to output json')

# Parse the command line arguments
args = parser.parse_args()

input_json_file = "./LongHisDoc.json"

output_json_file = args.output_path

ocr_dir = "./OCR_Res/"


RERANK_MODEL = args.model_path


tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(RERANK_MODEL).to(device)


import torch
import torch.nn.functional as F
from tqdm import tqdm
# from tqdm import tqdm

def encoder(texts):
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def find_top_k_similar(texts, query,index_ls, k=15, batch_size=8):
    # 处理文本列表，获取所有文本的 [CLS] 向量
    all_text_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = encoder(batch)
        all_text_vecs.append(embeddings)

    # 将所有批次的嵌入拼接在一起
    all_text_vecs = torch.cat(all_text_vecs, dim=0)  # Shape: [num_texts, hidden_size]

    # 对查询文本进行嵌入
    query_vec = encoder([query])  # Shape: [1, hidden_size]

    # 计算余弦相似度
    # 使用 F.cosine_similarity 来计算文本向量和查询向量的相似度
    # import pdb
    # pdb.set_trace()
    similarities = F.cosine_similarity(query_vec, all_text_vecs)  # Shape: [num_texts]

    # 获取相似度最高的前 k 个文本的索引
    if k >len(texts):
        k = len(texts)
    top_k_indices = torch.topk(similarities, k=k).indices

    # 获取对应的最相似文本
    top_k_texts = [texts[i] for i in top_k_indices]

    top_k_index = [index_ls[i] for i in top_k_indices] 

    return top_k_texts, top_k_index 






with open(input_json_file, 'r') as f:
    data = json.load(f)

for d_key, d_value in tqdm(data.items()):
  

    json_p = ocr_dir + str(d_key) + ".json"
    anno_dic = json.load(open(json_p))
    # import pdb
    # pdb.set_trace()
    texts = []
    index_ls = []
    for key, value in anno_dic.items():
        # if 'punc' in value:  # 检查是否有 punc 字段
        try:
            texts.append(value["punc"])
        except:
            texts.append("")
        index_ls.append(int(key) + 1)
    



    for d in tqdm(d_value):
        # txt_file = d["doc_id"]
        # CHROMADB_COLLECTION_NAME= f"{txt_file}_pages_embeddings"
        # vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)
        user_query = d["question"]
        top_k_texts, top_k_index =find_top_k_similar(texts, user_query,index_ls, k=50, batch_size=64)
        d["top_pages"] = top_k_index 
        # d["text_retriver_docs"] = top_k_texts

        with open(output_json_file, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)