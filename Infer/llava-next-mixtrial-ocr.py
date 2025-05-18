from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from tqdm import tqdm
import json
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt


model_path ="./Model/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_path)

model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,attn_implementation="flash_attention_2", device_map="auto") 




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
  
    inputs = processor( text=prompt, return_tensors="pt").to("cuda:0")

    output = model.generate(**inputs, max_new_tokens=1024)

    res = processor.decode(output[0], skip_special_tokens=True)

    res = res.split('[/INST]', 1)[-1]

    return res

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

        d['model_res'] = res
        print(res)
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)


