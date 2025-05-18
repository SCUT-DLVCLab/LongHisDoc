from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from tqdm import tqdm
import torch
from generate_prompt import text_system_prompt,text_generate_prompt,img_system_prompt,img_generate_prompt
model_name = "./Model/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def truncate_prompt_to_token_limit(prompt, max_prompt_tokens=32000, encoding_name=model_name):
    """
    截断 prompt 字符串，使其 token 数 <= max_prompt_tokens。
    返回截断后的 prompt 字符串。
    """
    enc = AutoTokenizer.from_pretrained(encoding_name)
    tokens = enc.encode(prompt)
    # import pdb
    # pdb.set_trace()
    if len(tokens) <= max_prompt_tokens:
        return prompt  # 无需截断

    truncated_tokens = tokens[:max_prompt_tokens]  # 截取结尾部分（保留末尾语义）
    truncated_prompt = enc.decode(truncated_tokens)
    return truncated_prompt

def communicate_with_qwen(prompt):
    
    messages=[
            {"role": "system", "content": text_system_prompt},
            {"role": "user", "content": prompt},
        ]
    # import pdb
    # pdb.set_trace()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # import pdb
    # pdb.set_trace()
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
        
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # print(response)

    return response


def read_json(json_p,page):
    anno_dic = json.load(open(json_p))
    
    punc_text=anno_dic[str(page)]['punc']
    # print(f"有异体字：{punc_text}")
    # simplified=convert(punc_text, 'zh-cn')
    # print(f"无异体字：{simplified}")

    return punc_text


# json_file_path = '/data1/szy/LLM_code/GujiDoc_collections/data_json_kv/merged_data_3_6_kv_all_fixed_final.json'
json_file_path = '/data1/szy/LLM_code/GujiDoc_collections/LCHD_benchmark_test_code/data/merged_data_3_26_kv_all_fixed_final_input_pages_book_name_fangbo.json'
# json_file_path = '/data1/szy/LLM_code/GujiDoc_collections/test_code/generator_infer/res/qwen2_5-vl/Qwen2.5-VL-7B-Instruct_with_evidence_ocr.json'

images_dir="/data1/szy/LLM_code/GujiDoc_collections/biaozhu_shuju_final_images_under_1024"

ocr_dir = "/data1/szy/LLM_code/GujiDoc_collections/guji_anno_3_6_ocr_res_final_sorted"

out_json_path = '/data1/szy/LLM_code/GujiDoc_collections/LCHD_benchmark_test_code/res/LLM_OCR/'+ model_name.split("/")[-1] +'_4_20_test.json'

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)



for d_key, d_value in tqdm(classified_data.items(), desc="Processing doc", leave=False):
    for d in tqdm(d_value, desc="Processing doc", leave=False):
        if "model_res" in d:
            continue
        doc_id = d["doc_id"]
        # if d["evidence_pages"]:
        # if "input_pages" in d and d["evidence_pages"]:
        page = json.loads(d["evidence_pages"])
        # page = d["top_pages"][:5]
        if isinstance(page,int):
            page = [page]
        page_input = json.loads(d["input_pages"])


        query = d["question"]

        ocr_res = ""
        doc_id = d["doc_id"]
        # page = json.loads(d["evidence_pages"]) if "evidence_pages" in d else []
        # page_input = json.loads(d["input_pages"]) if "input_pages" in d else []
        query = d["question"]
        ocr_res = ""
        ocr_json = f"{ocr_dir}/{doc_id}.json"
        
        anno_dic = json.load(open(ocr_json))

        for anno in anno_dic:
            if anno_dic[anno]['punc'] != "":
                ocr_res = ocr_res + f"第{int(anno)+1}页：\n"
                ocr_res = ocr_res + anno_dic[anno]['punc'] + "\n"

        # ocr_res = truncate_prompt_to_token_limit(ocr_res)


        prompt_ocr = text_generate_prompt.replace("{ocr_res}",ocr_res).replace('{query}',query)
        # import pdb
        # pdb.set_trace()
        
        
        # import pdb
        # pdb.set_trace()
        
        # print(image_path_ls)

        # response=communicate_with_openai(ocr_page_ls,prompt_image, True, image_path_ls)
        res = communicate_with_qwen(prompt_ocr)
        print(res)
        d['model_res'] = res
        with open(out_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(classified_data, json_file, ensure_ascii=False, indent=4)