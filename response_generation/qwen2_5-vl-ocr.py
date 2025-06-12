
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


from tqdm import tqdm
import json
from zhconv import convert
import os
import torch
import tiktoken
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt

# default: Load the model on the available device(s)

import argparse


parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="./Model/Qwen2.5-VL-7B-Instruct", help='Path to the model')

# Parse the command line arguments
args = parser.parse_args()

# Use the provided model path
model_path = args.model_path


def truncate_prompt_to_token_limit(prompt, max_prompt_tokens=127500, encoding_name="./Model/Qwen2.5-VL-7B-Instruct"):
    """
    截断 prompt 字符串，使其 token 数 <= max_prompt_tokens。
    返回截断后的 prompt 字符串。
    """
    enc = AutoTokenizer.from_pretrained(encoding_name)
    tokens = enc.encode(prompt)
   
    if len(tokens) <= max_prompt_tokens:
        return prompt  # 无需截断

    truncated_tokens = tokens[:max_prompt_tokens]  # 截取结尾部分（保留末尾语义）
    truncated_prompt = enc.decode(truncated_tokens)
    return truncated_prompt


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, device_map="auto",torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2"
)



processor = AutoProcessor.from_pretrained(model_path) 



def infer_qwen2vl_with_ocr(text):

    messages = [
    {
        "role": "user",
        "content": [
            
        ],
    }
    ]
    
    
    text_content={
            "type":"text",
            "text": text
        }
    messages[0]["content"].append(text_content)

    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
   
    
    inputs = processor(
        text=[text],
        padding=False,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024,do_sample=False)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])

    return output_text[0]


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
       


        ocr_res = truncate_prompt_to_token_limit(ocr_res)
        prompt_ocr = text_system_prompt +"\n"+ text_generate_prompt.replace('{ocr_res}',ocr_res).replace('{query}',query)
        # import pdb
        # pdb.set_trace()
        
        
        # import pdb
        # pdb.set_trace()
        
        # print(image_path_ls)

        # response=communicate_with_openai(ocr_page_ls,prompt_image, True, image_path_ls)
        res = infer_qwen2vl_with_ocr(prompt_ocr)
        # print(res)
        d['model_res'] = res
        torch.cuda.empty_cache()
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)
