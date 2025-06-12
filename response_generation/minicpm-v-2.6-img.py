import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt
import math
import argparse


parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="./Model/MiniCPM-V-2_6", help='Path to the model')

# Parse the command line arguments
args = parser.parse_args()

# Use the provided model path
model_path = args.model_path

model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16) 
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def concat_images(image_list, concat_num=5, column_num=3):
    interval = max(math.ceil(len(image_list) / concat_num), 1)
    concatenated_image_list = list()

    for i in range(0, len(image_list), interval):
       
        images_this_batch = [
            Image.open(filename) for filename in image_list[i:i + interval]
        ]
        if column_num == 1:
            total_height = images_this_batch[0].height * len(images_this_batch)
        else:
            total_height = images_this_batch[0].height * ((len(images_this_batch) - 1) // column_num + 1)

        concatenated_image = Image.new('RGB', (images_this_batch[0].width * column_num, total_height), 'white')
        x_offset, y_offset = 0, 0
        for cnt, image in enumerate(images_this_batch):
            concatenated_image.paste(image, (x_offset, y_offset))
            x_offset += image.width
            if (cnt + 1) % column_num == 0:
                y_offset += image.height
                x_offset = 0
        
        concatenated_image_list.append(concatenated_image)

    return concatenated_image_list



def infer_minicpm_with_image(prompt_image,image_path_ls):
    
    input_ls = concat_images(image_path_ls)
   
    input_ls.append(prompt_image)
    

    msgs = [{'role': 'user', 'content': input_ls}]

    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        do_sample=False
        
    )
    print(answer)
    return answer



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
        # if d["evidence_pages"]:
        # if "input_pages" in d and d["evidence_pages"]:
        # page = json.loads(d["top_pages"])[:5]
        # page = json.loads(d["evidence_pages"])
        # if isinstance(page,int):
        #     page = [page]
        page_input = json.loads(d["input_pages"])


        query = d["question"]

        
        prompt_image= img_system_prompt +"\n"+ img_generate_prompt.replace('{query}',query)
        # import pdb
        # pdb.set_trace()
        image_path_ls=[]
        for p in page_input:
            # 
            image_path= images_dir + "/" + doc_id +"/" + f"page_{p}.jpg"
            # image_path=find_image(images_dir,doc_id,p)
            image_path_ls.append(image_path)
        
        # import pdb
        # pdb.set_trace()
        
        # print(image_path_ls)

        # response=communicate_with_openai(ocr_page_ls,prompt_image, True, image_path_ls)
        res = infer_minicpm_with_image(prompt_image,image_path_ls)
        # print(res)
        d['model_res'] = res
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)