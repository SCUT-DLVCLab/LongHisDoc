import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import json
from PIL import Image
import math
from tqdm import tqdm
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt
import argparse

def concat_images(image_list, concat_num=1, column_num=3):
    interval = max(math.ceil(len(image_list) / concat_num), 1)


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

    return concatenated_image


device = "cuda"
parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="./Model/glm-4v-9b", help='Path to the model')

# Parse the command line arguments
args = parser.parse_args()

# Use the provided model path
model_path = args.model_path

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
).to(device).eval()


def chat_with_sp_model(text,image_ls):
    image = concat_images(image_ls, concat_num=1, column_num=3)



    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": text}],
                                        add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                        return_dict=True)  # chat mode

    inputs = inputs.to(device)


    gen_kwargs = {"max_length": 1024, "do_sample": False, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


json_file_path = './LongHisDoc.json'

images_dir = "./LongHisDoc_IMG"

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

        prompt_image= img_system_prompt +"\n"+ img_generate_prompt.replace('{query}',query)
       
        image_path_ls=[]
     
        for p in page_input:
             
            image_path= images_dir + "/" + doc_id +"/" + f"page_{p}.jpg"
            
            image_path_ls.append(image_path)
       
        res = chat_with_sp_model(prompt_image,image_path_ls)
       
        d['model_res'] = res
        print(res)
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)





