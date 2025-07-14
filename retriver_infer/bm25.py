import json
import jieba


import numpy as np
from rank_bm25 import BM25Okapi



import torch
import torch.nn.functional as F
from tqdm import tqdm
from zhconv import convert



def find_top_k_similar(texts, query, index_ls, k=15):
    

    texts =[convert(text, 'zh-cn') for text in texts]
    query = convert(query, 'zh-cn')
    tokenized_corpus = [list(filter(lambda x: x != "", list(jieba.cut(doc)))) for doc in texts]
    tokenized_corpus= [[item.lower() for item in sublist] for sublist in tokenized_corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    
    
    tokenized_query = [item.lower() for item in list(jieba.cut(query))]

    

    doc_scores = bm25.get_scores(tokenized_query)
    
    if k >len(texts):
        k = len(texts)

    
    top_k_indices = np.argsort(doc_scores)[-k:][::-1]

    top_k_texts = [texts[i] for i in top_k_indices]

    top_k_index = [index_ls[i] for i in top_k_indices] 

    return top_k_texts, top_k_index 


import argparse

parser = argparse.ArgumentParser(description="Receive the model path as an external input")

parser.add_argument('--model_path', type=str, default="bm25", help='Path to the model')

parser.add_argument('--output_path', type=str, default="./retrivers_res/bm25.json", help='Path to output json')

args = parser.parse_args()

model_name = args.model_path

input_json_file = "./LongHisDoc.json"

output_json_file = args.output_path

ocr_dir = "./OCR_Res/"

with open(input_json_file, 'r') as f:
    data = json.load(f)

for d_key, d_value in tqdm(data.items()):
    

    json_p = ocr_dir + str(d_key) + ".json"
    anno_dic = json.load(open(json_p))
    texts = []
    index_ls = []
   
    
    for key, value in anno_dic.items():
        
        texts.append(value["punc"])
        index_ls.append(str(int(key)+1))
    
   


    for d in tqdm(d_value):
        
        user_query = d["question"]

        top_k_texts, top_k_index = find_top_k_similar(texts, user_query, index_ls, k=50)
        
        d["top_pages"] = top_k_index 

        with open(output_json_file, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)


