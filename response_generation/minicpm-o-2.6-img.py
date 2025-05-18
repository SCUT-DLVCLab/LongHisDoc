import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt
model_path = "./Model/MiniCPM-o-2_6"



model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation='flash_attention_2', 
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio = False,
    init_tts = False
)


model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# In addition to vision-only mode, tts processor and vocos also needs to be initialized
model.init_tts()



def infer_minicpm_with_image(prompt_image,image_path_ls):
    input_ls = []
    for image in image_path_ls:
        page_num = int(image.replace(".jpg","").split("_")[-1])
        input_ls.append(f"第{page_num}页：\n")
        input_ls.append(Image.open(image).convert('RGB'))

    input_ls.append(prompt_image)
    

    msgs = [{'role': 'user', 'content': input_ls}]

    answer = model.chat(
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
      
        page_input = json.loads(d["input_pages"])


        query = d["question"]

        
        prompt_image= img_system_prompt +"\n"+ img_generate_prompt.replace('{query}',query)
        
        image_path_ls=[]
        for p in page_input:
             
            image_path= images_dir + "/" + doc_id +"/" + f"page_{p}.jpg"
            
            image_path_ls.append(image_path)
        
       
        res = infer_minicpm_with_image(prompt_image,image_path_ls)
       
        d['model_res'] = res
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)