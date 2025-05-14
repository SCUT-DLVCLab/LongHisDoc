import requests
import json
from tqdm import tqdm
from openai import OpenAI
from zhconv import convert
import os
import concurrent.futures
from all_extracted_prompt import extracted_Str_prompt,extracted_Bool_prompt,extracted_List_prompt,extracted_Enum_prompt,extracted_None_prompt,extracted_Int_prompt

import threading
# 创建线程锁用于文件写入
lock = threading.Lock()
def custom_trad_to_simp(text):
    mapping = {"𡊮": "袁","叅":"参"}
    for trad, simp in mapping.items():
        text = text.replace(trad, simp)
    text = convert(text, 'zh-cn')
    return text

# client = OpenAI(api_key="sk-nUt8NbQS7BccFTBCVSsUPuOy8JreNOfm3PSxPxE2tMhcJ3eo", base_url="https://api.nuwaapi.com/v1")

client = OpenAI(api_key="sk-S6AgxYu8HVTEisDkwMljzusx7eVA6xpUldxBF9vmVBu1R5p9", base_url="https://api.nuwaapi.com/v1")

MODEL_NAME = "gpt-4o-2024-11-20"

def communicate_with_openai(prompt):
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.json()

def process_item(d):
    if "LLM_scored" in d and d["LLM_scored"]!=None:
        return
    # if d["model_res"]=="######发生错误#######":

    # question = custom_trad_to_simp(d["question"])
    # answer = custom_trad_to_simp(d["answer"])
    # model_res = custom_trad_to_simp(d["model_res"])

    # answer_type = d["answer_type"]
        
    try:
        if d["answer_type"] == "Str":
            question = custom_trad_to_simp(d["question"])
            
            model_res = custom_trad_to_simp(str(d["model_res"]))

           

            score_prompt=extracted_Str_prompt.replace('{question}', question).replace('{model_res}', model_res)   
            
        elif d["answer_type"] == "Int":
            question = custom_trad_to_simp(d["question"])
            model_res = custom_trad_to_simp(str(d["model_res"]))

           

            score_prompt=extracted_Int_prompt.replace('{question}', question).replace('{model_res}', model_res)

        elif d["answer_type"] == "Bool":
            question = custom_trad_to_simp(d["question"])
           
            model_res = custom_trad_to_simp(str(d["model_res"]))


            score_prompt=extracted_Bool_prompt.replace('{question}', question).replace('{model_res}', model_res)  

        elif d["answer_type"] == "Enum":
            question = custom_trad_to_simp(d["question"])
            
            model_res = custom_trad_to_simp(str(d["model_res"]))


            
            score_prompt = extracted_Enum_prompt.replace('{question}', question).replace('{model_res}', model_res)  

        elif d["answer_type"] == "List":
            question = custom_trad_to_simp(d["question"])
           
            model_res = custom_trad_to_simp(str(d["model_res"]))


           

            score_prompt = extracted_List_prompt.replace('{question}', question).replace('{model_res}', model_res) 


        elif d["answer_type"] == "None":
            question = custom_trad_to_simp(d["question"])
            
            model_res = custom_trad_to_simp(str(d["model_res"]))


            score_prompt=extracted_None_prompt.replace('{model_res}', model_res)

            # score_prompt=extracted_None_prompt
        # import pdb
        # pdb.set_trace()
        response = json.loads(communicate_with_openai(score_prompt))
        res = response["choices"][0]['message']['content']
        # d["LLM_scored"] = response["choices"][0]['message']['content']
        with lock:
            d["LLM_scored"] = res
            with open(out_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(classified_data, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"处理失败: {str(e)}")
        # d["LLM_scored"] = "处理失败"

# 文件路径配置保持不变
json_file_path = '/data1/szy/LLM_code/GujiDoc_collections/LCHD_benchmark_test_code/evidence_input/Qwen2.5-VL-7B-Instruct_evidence_5_10.json'
out_json_path = '/data1/szy/LLM_code/GujiDoc_collections/LCHD_benchmark_test_code/evidence_input/Qwen2.5-VL-7B-Instruct_evidence_5_10.json'

# 读取数据
with open(json_file_path, 'r', encoding='utf-8') as f:
    classified_data = json.load(f)

# 收集所有需要处理的任务
tasks = []
for d_key, d_value in classified_data.items():
    for d in d_value:
        if "model_res" not in d:
            pass
        elif d["model_res"]!="########输入过长###########":
            # tasks.append(d)
            if "LLM_scored" not in d:
                tasks.append(d)
            elif d["LLM_scored"] == None:
                tasks.append(d)
        # el
# 使用线程池处理
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:  # 可根据API限制调整线程数
    futures = [executor.submit(process_item, task) for task in tasks]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        future.result()  # 等待所有任务完成
        pass  # 这里只等待任务完成，不处理返回结果

# 最终保存结果
# with open(out_json_path, 'w', encoding='utf-8') as f:
#     json.dump(classified_data, f, ensure_ascii=False, indent=4)