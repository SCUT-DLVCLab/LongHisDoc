import torch
import os
from transformers import AutoTokenizer, AutoModel
from icecream import ic
import time
from tqdm import tqdm
import json
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt
from PIL import Image
import math

def concat_images(image_list, concat_num=10, column_num=3):
    interval = max(math.ceil(len(image_list) / concat_num), 1)
    concatenated_image_list = list()

    for i in range(0, len(image_list), interval):
        # 拼接图像
        images_this_batch = [
            Image.open(filename) for filename in image_list[i:i + interval]
        ]
        if column_num == 1:
            total_height = images_this_batch[0].height * len(images_this_batch)
        else:
            total_height = images_this_batch[0].height * ((len(images_this_batch) - 1) // column_num + 1)

        concatenated_image = Image.new('RGB', (images_this_batch[0].width * column_num, total_height), 'white')
        x_offset, y_offset = 0, 0
        for cnt, image in enumerate(images_this_batch):
            concatenated_image.paste(image, (x_offset, y_offset))
            x_offset += image.width
            if (cnt + 1) % column_num == 0:
                y_offset += image.height
                x_offset = 0
        
        concatenated_image_list.append(concatenated_image)

    return concatenated_image_list

class DocOwlInfer():
    def __init__(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False,local_files_only=True,attn_implementation="flash_attention_2")
        self.model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=504, crop_anchors='grid_12')
        
    def inference(self, images, query):
        messages = [{'role': 'USER', 'content': '<|image|>'*len(images)+query}]
        answer = self.model.chat(messages=messages, images=images, tokenizer=self.tokenizer)
        return answer

model_path = './Model/DocOwl2'
docowl = DocOwlInfer(ckpt_path=model_path)



json_file_path = './LongHisDoc.json'

images_dir="./LongHisDoc"

out_json_path = './infer_res/' + model_path.split("/")[-1] +'_img.json'

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)


    

for d_key, d_value in tqdm(classified_data.items(), desc="Processing doc", leave=False):
    for d in tqdm(d_value, desc="Processing doc", leave=False):
        if "model_res" in d:
            continue
        doc_id = d["doc_id"]
       
        page_input = json.loads(d["input_pages"])
       

        query = d["question"]

        prompt_image= img_system_prompt +"\n"+ img_generate_prompt.replace('{query}',query)
        
        image_path_ls=[]
        
        for p in page_input:
             
            image_path= images_dir + "/" + doc_id +"/" + f"page_{p}.jpg"
          
            image_path_ls.append(image_path)
            
        print(image_path_ls)
        merge_img_ls = concat_images(image_path_ls, concat_num=10, column_num=3)
        res = docowl.inference(merge_img_ls, query=prompt_image)

        print(res)
        d['model_res'] = res
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)


