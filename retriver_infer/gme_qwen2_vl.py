# You can find the script gme_inference.py in https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py
from gme_inference import GmeQwen2VL
import json
import os
from tqdm import tqdm
import heapq
import torch
import re




import argparse

parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="gme_qwen2_vl", help='Path to the model')

parser.add_argument('--output_path', type=str, default="./retrivers_res/gme_qwen2_vl.json", help='Path to output json')

# Parse the command line arguments
args = parser.parse_args()


json_file_path = "./LongHisDoc.json"


model_name = args.model_path


out_json_path= args.output_path


image_directory = './LongHisDoc_IMG'  


gme = GmeQwen2VL(model_name)


batch_size = 8  # 设置批量大小


# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)


# Single-modal embedding
# e_text = gme.get_text_embeddings(texts=texts)
# e_image = gme.get_image_embeddings(images=images)
# print((e_text * e_image).sum(-1))



for root, dirs, files in os.walk(image_directory):
    for folder_name in dirs:
        folder_path = os.path.join(root, folder_name)
        # 只处理存在于doc_ids中的文件夹
        if os.path.isdir(folder_path) and folder_name in classified_data:
            # 获取图片路径
            image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]

            # 从JSON中获取样本列表
            samples = classified_data[folder_name]

            # 批量处理所有图片
            all_embeddings = []
            # 读取图片并放入 passages 中
            
            print("Downloading images...")
            passages = []
            # import pdb
            # pdb.set_trace()
            # for img_path in image_paths:
                
            #     img = Image.open(img_path)
            #     passages.append(img)
            # print("Images downloaded.")

            # batch_size = 4  # 每个批次处理的图片数量


            for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images", leave=False):
                # batch_image_paths = image_paths[i:i + batch_size]
                # batch_images_processed = {key: tensor.to(model.device) for key, tensor in batch_images_processed.items()}
                # batch_images_processed = batches_images_processed[i:i + batch_size]  # 读取当前批量的图片
                batch_image_paths = image_paths[i:i + batch_size]  # 获取当前批次的图片路径
                # 处理输入
                # batch_images_processed = processor.process_images(batch_images).to(model.device)

                # 前向传播
                # import pdb
                # pdb.set_trace()
                with torch.no_grad():
                    image_embeddings = gme.get_image_embeddings(images=batch_image_paths)
                    # if image_embeddings.shape[1]==759:
                        # import pdb
                        # pdb.set_trace()
                        # print(batch_image_paths)

                all_embeddings.append(image_embeddings)

            # 合并所有图片的embedding
            # import pdb
            # pdb.set_trace()
            all_embeddings = torch.cat(all_embeddings)
            # import pdb
            # pdb.set_trace()
            # with torch.no_grad():
            #     e_image = gme.get_image_embeddings(images=image_paths)
            

            # 逐个处理样本
            for sample in tqdm(samples, desc=f"Processing samples in {folder_name}", leave=False):
                question = sample['question']  # 读取问题
                
                queries = [question]
                # batch_queries = processor.process_queries([question]).to(model.device)  # 处理当前问题

                # 前向传播
                with torch.no_grad():
                    e_text =  gme.get_text_embeddings(texts=queries)
                scores = (e_text * all_embeddings).sum(-1)
                score_ls=scores.tolist()
                # import pdb
                # pdb.set_trace()
                # # print(score_ls)
                # score_ls_1=score_ls[0]
                top_k_indices = [index for index, value in heapq.nlargest(50, enumerate(score_ls), key=lambda x: x[1])]
                

                top_k_files = [image_paths[idx] for idx in top_k_indices]

                top_page = [
                    int(re.search(r'page_(\d+)', os.path.basename(path)).group(1)) 
                    for path in top_k_files
                ]

                sample['top_pages']=top_page
                # print(f"Top_{len(top_k_files)}:{top_k_files}")

                with open(out_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(classified_data, json_file, ensure_ascii=False, indent=4)
