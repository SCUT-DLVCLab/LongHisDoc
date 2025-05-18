import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import json
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt




model_id = "./Model/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)


def infer_llm_with_ocr(text):

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
    

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(text=prompt, return_tensors="pt").to("cuda:0")

    output = model.generate(**inputs, max_new_tokens=1024,do_sample=False)

    res = processor.decode(output[0], skip_special_tokens=True)

    res = res.split("assistant")[-1]

    return res

json_file_path = './LongHisDoc.json'

ocr_dir = "./OCR_Res"

out_json_path = './infer_res/Llama-3.2-11B-Vision-Instruct_ocr.json'

    

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

        ocr_res = ""
        ocr_json = f"{ocr_dir}/{doc_id}.json"
        
        anno_dic = json.load(open(ocr_json))

        for anno in anno_dic:
            if anno_dic[anno]['punc'] != "":
                ocr_res = ocr_res + f"第{int(anno)+1}页：\n"
                ocr_res = ocr_res + anno_dic[anno]['punc'] + "\n"
       


        prompt_ocr = text_system_prompt +"\n"+ text_generate_prompt.replace('{ocr_res}',ocr_res).replace('{query}',query)
    
        res = infer_llm_with_ocr(prompt_ocr)
        print(res)
        d['model_res'] = res
        
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)


