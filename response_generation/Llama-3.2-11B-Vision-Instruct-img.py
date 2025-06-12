import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import json
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt

import argparse

parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default='./Model/Llama-3.2-11B-Vision-Instruct', help='Path to the model')

# Parse the command line arguments
args = parser.parse_args()

# Use the provided model path
model_path = args.model_path


model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)


def infer_llm_with_image(text,image_ls):

    messages = [
    {
        "role": "user",
        "content": [
        ],
    }
    ]

    for image in image_ls:
        page_num = int(image.replace(".jpg","").split("_")[-1])
        text_content={
            "type":"text",
            "text": f"第{page_num}页：\n"
        }
        messages[0]["content"].append(text_content)
        image_content={
            "type":"image",
        }
        messages[0]["content"].append(image_content)

    text_content={
            "type":"text",
            "text": text
        }
    messages[0]["content"].append(text_content)
    image_input = [Image.open(image_path) for image_path in image_ls]

 

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(images=image_input, text=prompt, return_tensors="pt").to("cuda:0")

   
    output = model.generate(**inputs, max_new_tokens=1024,do_sample=False)

    res = processor.decode(output[0], skip_special_tokens=True)

    res = res.split("assistant")[-1]

    return res


json_file_path = './LongHisDoc.json'

images_dir="./LongHisDoc_IMG"

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

        prompt_image = img_system_prompt +"\n"+ img_generate_prompt.replace('{query}',query)
        
        image_path_ls=[]
       
        for p in page_input:
            
            image_path= images_dir + "/" + doc_id +"/" + f"page_{p}.jpg"
            
            image_path_ls.append(image_path)
            
     
        res = infer_llm_with_image(prompt_image,image_path_ls)
        print(res)
        d['model_res'] = res
       
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)

