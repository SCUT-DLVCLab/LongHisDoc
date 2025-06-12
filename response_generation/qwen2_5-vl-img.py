
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


from tqdm import tqdm
import json
import torch
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt

import argparse


parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="./Model/Qwen2.5-VL-7B-Instruct", help='Path to the model')

# Parse the command line arguments
args = parser.parse_args()

# Use the provided model path
model_path = args.model_path



model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, device_map="auto",torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
)

# default processer
processor = AutoProcessor.from_pretrained(model_path) 


def infer_qwen2vl_with_image(text,image_ls):

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
            "image": image
        }
        messages[0]["content"].append(image_content)

    text_content={
            "type":"text",
            "text": text
        }
    messages[0]["content"].append(text_content)


    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
   
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
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

images_dir="./LongHisDoc_IMG"


out_json_path = './infer_res/' + model_path.split("/")[-1] +'.json'

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
        # if page:
        for p in page_input:
            
            image_path= images_dir + "/" + doc_id +"/" + f"page_{p}.jpg"
           
            image_path_ls.append(image_path)
       
        res = infer_qwen2vl_with_image(prompt_image,image_path_ls)
        
        d['model_res'] = res
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)
