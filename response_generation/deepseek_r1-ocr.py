import openai
from openai import OpenAI

import json
from tqdm import tqdm

from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2VLForConditionalGeneration

client = OpenAI(api_key="", base_url="")

def truncate_prompt_to_token_limit(prompt, max_prompt_tokens=64000, encoding_name="./Model/deepseek_tokenizer"):
    """
    截断 prompt 字符串，使其 token 数 <= max_prompt_tokens。
    返回截断后的 prompt 字符串。
    """
    enc = AutoTokenizer.from_pretrained(encoding_name)
    
    tokens = enc.encode(prompt)
    
    if len(tokens) <= max_prompt_tokens:
        return prompt  

    truncated_tokens = tokens[:max_prompt_tokens]  
    truncated_prompt = enc.decode(truncated_tokens)
    return truncated_prompt

def communicate_with_deepseek(prompt):
    
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": text_system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.0,
        max_tokens=1024

    )

    return response.choices[0].message.content


def read_json(json_p,page):
    anno_dic = json.load(open(json_p))
    
    punc_text=anno_dic[str(page)]['punc']
    

    return punc_text

def read_json_anno_dic(json_p):
    anno_dic = json.load(open(json_p))
    return anno_dic

json_file_path = './LongHisDoc.json'


ocr_dir = "./OCR_Res"

out_json_path = './infer_res/Deepseek-r1-api.json'

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)



for d_key, d_value in tqdm(classified_data.items(), desc="Processing doc", leave=False):
    for d in tqdm(d_value, desc="Processing doc", leave=False):
        if "model_res" in d and d["model_res"]!="######输入过长#######":
            continue
        doc_id = d["doc_id"]
        
        page = json.loads(d["evidence_pages"])
        
        if isinstance(page,int):
            page = [page]
        page_input = json.loads(d["input_pages"])


        query = d["question"]

        ocr_res = ""
      
        ocr_json= ocr_dir + "/" + doc_id + ".json"
        anno_dic = read_json_anno_dic(ocr_json)
            
        for anno in anno_dic:

                
            if anno_dic[anno]['punc'] != "":
                ocr_res = ocr_res + f"第{int(anno)+1}页：\n"
                ocr_res = ocr_res + anno_dic[anno]['punc'] + "\n"
                
                
        



        prompt_ocr = text_generate_prompt.replace("{ocr_res}",ocr_res).replace('{query}',query)
        

        try:
            res = communicate_with_deepseek(prompt_ocr)
            d['model_res'] = res
            print(res)
        except openai.BadRequestError as e:
            if "maximum context length" in str(e):
                print(f"######输入过长#######: {e}")
                ocr_res = truncate_prompt_to_token_limit(ocr_res, max_prompt_tokens=64000)
                prompt_ocr_chunk = text_generate_prompt.replace("{ocr_res}",ocr_res).replace('{query}',query)
                res = communicate_with_deepseek(prompt_ocr_chunk)
                d['model_res'] = res
                
                print(res)
                
               
            else:
                
                print(f"发生其他请求错误: {e}")
        
        
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)

