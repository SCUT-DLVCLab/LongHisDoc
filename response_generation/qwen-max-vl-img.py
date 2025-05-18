from openai import OpenAI
from openai import OpenAI
import json
import base64
import argparse
import fitz

from PIL import Image
from tqdm import tqdm
import concurrent.futures
import threading
import os
import openai
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt


client = OpenAI(api_key="", base_url="")

MODEL_NAME = "qwen-vl-max"
# 创建线程锁用于文件写入
lock = threading.Lock()





def communicate_with_llm(prompt,img_path_ls):
    def encode_image(image_path):
        
        with open(image_path, "rb") as image_file:
          return base64.b64encode(image_file.read()).decode('utf-8')
    

    content = list()
    
     
    for img_path in img_path_ls:
        encoded_image = encode_image(img_path)
        page_num = int(img_path.replace(".jpg","").split("_")[-1])
        
        content.append({
            "type": "text",
            "text": f"第{page_num}页：\n",
        })

        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
   
    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": img_system_prompt},
            {"role": "user", "content": content},
        ],
        stream=False,
        temperature=0.0,
        max_tokens=1024
    )
    return response.choices[0].message.content

def read_json(json_p, page):
    anno_dic = json.load(open(json_p))
    return anno_dic[str(page)]['punc']



def process_single_d(d, images_dir):
    try:
        doc_id = d["doc_id"]
        
        query = d["question"]
        
        page_input = json.loads(d["input_pages"])
        
        

        prompt_image= img_generate_prompt.replace('{query}',query)


        image_path_ls=[]

        for p in page_input:
            
            image_path= images_dir + "/" + doc_id +"/" + f"page_{p}.jpg"
            
            image_path_ls.append(image_path)

        try:
            
            res = communicate_with_llm(prompt_image,image_path_ls)
            print(res)
           
        except openai.BadRequestError as e:
            if "maximum context length" in str(e):
                
                res = "######输入过长#######"
                
            else:
                
                print(f"发生其他请求错误: {e}")
        
        
        # 使用锁更新结果并保存
        with lock:
            d['model_res'] = res
            
            with open(out_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(classified_data, json_file, ensure_ascii=False, indent=4)
                
    except Exception as e:
        print(f"Error processing {doc_id}: {str(e)}")

# 配置路径


json_file_path = './LongHisDoc.json'

images_dir="./LongHisDoc_IMG"

out_json_path = './infer_res/' + MODEL_NAME +'_img.json'

# 加载数据
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)

# 收集需要处理的所有d条目
all_ds = []
for d_key, d_value in classified_data.items():
    for d in d_value:
        if "model_res" not in d:
            # print(d["doc_id"])
            all_ds.append(d)

# 创建线程池并发处理
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 可根据API限制调整线程数
    futures = []
    for d in all_ds:
        futures.append(executor.submit(process_single_d, d, images_dir))
    
    # 使用tqdm显示进度
    for future in tqdm(concurrent.futures.as_completed(futures), 
                      total=len(futures), 
                      desc="Processing"):
        future.result()  # 等待所有任务完成

print("All tasks completed!")
