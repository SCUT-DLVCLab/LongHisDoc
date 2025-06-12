import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt
import json

from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default='./Model/InternLm3-8B', help='Path to the model')

# Parse the command line arguments
args = parser.parse_args()

# Use the provided model path
model_path = args.model_path


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,device_map="auto",
    attn_implementation="flash_attention_2").cuda()

model = model.eval()

def truncate_prompt_to_token_limit(prompt, max_prompt_tokens=32000, encoding_name="./Model/internlm3-8b-instruct"):
    """
    截断 prompt 字符串，使其 token 数 <= max_prompt_tokens。
    返回截断后的 prompt 字符串。
    """
    enc = AutoTokenizer.from_pretrained(encoding_name, trust_remote_code=True)
    tokens = enc.encode(prompt)
    
    if len(tokens) <= max_prompt_tokens:
        return prompt  # 无需截断

    truncated_tokens = tokens[:max_prompt_tokens]  # 截断
    truncated_prompt = enc.decode(truncated_tokens)
    return truncated_prompt


def communicate_with_internlm3(prompt):
    messages = [
        {"role": "system", "content": text_system_prompt},
        {"role": "user", "content": prompt},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    generated_ids = model.generate(tokenized_chat, max_new_tokens=1024,do_sample=False)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
    ]
    prompt = tokenizer.batch_decode(tokenized_chat)[0]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response




def read_json(json_p,page):
    anno_dic = json.load(open(json_p))
    
    punc_text=anno_dic[str(page)]['punc']
    

    return punc_text

json_file_path = './LongHisDoc.json'

ocr_dir = "./OCR_Res"

out_json_path = './infer_res/' + model_path.split("/")[-1]  +'_ocr.json'

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)



for d_key, d_value in tqdm(classified_data.items(), desc="Processing doc", leave=False):
    for d in tqdm(d_value, desc="Processing doc", leave=False):
        if "model_res" in d:
            continue
        doc_id = d["doc_id"]
        
        page = json.loads(d["evidence_pages"])
        
        if isinstance(page,int):
            page = [page]
        page_input = json.loads(d["input_pages"])


        query = d["question"]

        doc_id = d["doc_id"]
        
        query = d["question"]
        ocr_res = ""
        ocr_json = f"{ocr_dir}/{doc_id}.json"
        
        anno_dic = json.load(open(ocr_json))

        for anno in anno_dic:
            if anno_dic[anno]['punc'] != "":
                ocr_res = ocr_res + f"第{int(anno)+1}页：\n"
                ocr_res = ocr_res + anno_dic[anno]['punc'] + "\n"

        ocr_res = truncate_prompt_to_token_limit(ocr_res)


        prompt_ocr = text_generate_prompt.replace("{ocr_res}",ocr_res).replace('{query}',query)
        
        res = communicate_with_internlm3(prompt_ocr)
        print(res)
        d['model_res'] = res
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)