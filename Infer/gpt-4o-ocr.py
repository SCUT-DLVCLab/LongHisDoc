from openai import OpenAI
from openai import OpenAI
import json
from tqdm import tqdm
import concurrent.futures
import threading
import openai
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt

client = OpenAI(api_key="", base_url="")

MODEL_NAME = "gpt-4o-2024-11-20"
# 创建线程锁用于文件写入
lock = threading.Lock()

def communicate_with_gemini(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": text_system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.0,
        max_tokens=1024
    )
    return response.choices[0].message.content

def read_json(json_p, page):
    anno_dic = json.load(open(json_p))
    return anno_dic[str(page)]['punc']



def process_single_d(d, ocr_dir):
    try:
        doc_id = d["doc_id"]

        query = d["question"]
        ocr_res = ""
        ocr_json = f"{ocr_dir}/{doc_id}.json"
        
        anno_dic = json.load(open(ocr_json))

        for anno in anno_dic:
            if anno_dic[anno]['punc'] != "":
                ocr_res = ocr_res + f"第{int(anno)+1}页：\n"
                ocr_res = ocr_res + anno_dic[anno]['punc'] + "\n"
       
        prompt_ocr= text_generate_prompt.replace("{ocr_res}",ocr_res).replace('{query}',query)
        
        # 调用API

        try:
          
            res = communicate_with_gemini(prompt_ocr)
            print(res)
            
        except openai.BadRequestError as e:
            if "maximum context length" in str(e):
                
                res = "######输入过长#######"
                
            else:
                
                print(f"发生其他请求错误: {e}")
        
        
        
        with lock:
            d['model_res'] = res
            if res== "######输入过长#######":
                d['LLM_scored'] = "0"
                d['score'] = 0.0
            with open(out_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(classified_data, json_file, ensure_ascii=False, indent=4)
                
    except Exception as e:
        print(f"Error processing {doc_id}: {str(e)}")

# 配置路径

json_file_path = './LongHisDoc.json'

ocr_dir = "./OCR_Res"

out_json_path = './infer_res/' + MODEL_NAME +'_ocr.json'

# 加载数据
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)

# 收集需要处理的所有d条目
all_ds = []
for d_key, d_value in classified_data.items():
    for d in d_value:
        if "model_res" not in d:
            all_ds.append(d)

# 创建线程池并发处理
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:  # 可根据API限制调整线程数
    futures = []
    for d in all_ds:
        futures.append(executor.submit(process_single_d, d, ocr_dir))
    
    # 使用tqdm显示进度
    for future in tqdm(concurrent.futures.as_completed(futures), 
                      total=len(futures), 
                      desc="Processing"):
        future.result()  # 等待所有任务完成

print("All tasks completed!")
