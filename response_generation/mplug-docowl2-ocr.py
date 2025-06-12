import torch
import os
from transformers import AutoTokenizer, AutoModel
from icecream import ic
import time
from tqdm import tqdm
import json
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt

class DocOwlInfer():
    def __init__(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False,local_files_only=True,attn_implementation="flash_attention_2")
        self.model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=504, crop_anchors='grid_12')
        
    def inference(self, query):
        messages = [{'role': 'USER', 'content': query}]
        answer = self.model.chat_ocr(messages=messages, tokenizer=self.tokenizer)
        return answer

import argparse


parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default='./Model/DocOwl2', help='Path to the model')

# Parse the command line arguments
args = parser.parse_args()

# Use the provided model path
model_path = args.model_path

docowl = DocOwlInfer(ckpt_path=model_path)


json_file_path = './LongHisDoc.json'

ocr_dir = "./OCR_Res"

out_json_path = './infer_res/' + model_path.split("/")[-1] +'_ocr.json'

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)

    

for d_key, d_value in tqdm(classified_data.items(), desc="Processing doc", leave=False):
    for d in tqdm(d_value, desc="Processing doc", leave=False):
        if "model_res" in d:
            continue
        doc_id = d["doc_id"]
        # if d["evidence_pages"]:
        # if "input_pages" in d and d["evidence_pages"]:
        page = json.loads(d["evidence_pages"])
        # page = d["top_pages"][:5]
        if isinstance(page,int):
            page = [page]
        page_input = json.loads(d["input_pages"])


        query = d["question"]

        # ocr_res = ""
        ocr_res = ""
        ocr_json = f"{ocr_dir}/{doc_id}.json"
        
        anno_dic = json.load(open(ocr_json))

        for anno in anno_dic:
            if anno_dic[anno]['punc'] != "":
                ocr_res = ocr_res + f"第{int(anno)+1}页：\n"
                ocr_res = ocr_res + anno_dic[anno]['punc'] + "\n"
       


        # ocr_res = truncate_prompt_to_token_limit(ocr_res)
        prompt_ocr = text_system_prompt +"\n"+ text_generate_prompt.replace('{ocr_res}',ocr_res).replace('{query}',query)
        # import pdb
        # pdb.set_trace()
        
        
        # import pdb
        # pdb.set_trace()
        
        # print(image_path_ls)

        # response=communicate_with_openai(ocr_page_ls,prompt_image, True, image_path_ls)
        image=[]
        res = docowl.inference(prompt_ocr)
        # print(res)
        d['model_res'] = res
        # torch.cuda.empty_cache()
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)


