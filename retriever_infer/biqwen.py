import os
import json
import torch
from PIL import Image
import sys
# from colpali_engine.models import ColQwen2, ColQwen2Processor
# from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from tqdm import tqdm
import re
# from colpali_engine.models import ColPali, ColPaliProcessor
import torch
from PIL import Image

from colpali_engine.models import BiQwen2, BiQwen2Processor


import argparse

parser = argparse.ArgumentParser(description="Receive the model path as an external input")

# Add the model path argument
parser.add_argument('--model_path', type=str, default="biqwen", help='Path to the model')

parser.add_argument('--output_path', type=str, default="./retrivers_res/biqwen.json", help='Path to output json')

# Parse the command line arguments
args = parser.parse_args()


json_file_path = "./LongHisDoc.json"
image_directory = './LongHisDoc_IMG'  # 图片文件夹路径

model_name = args.model_path


out_json_path= args.output_path

batch_size = 1  # 设置批量大小

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    classified_data = json.load(json_file)



model = BiQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",  # or "mps" if on Apple Silicon
).eval()
processor = BiQwen2Processor.from_pretrained(model_name)


for root, dirs, files in os.walk(image_directory):
    for folder_name in tqdm(dirs, desc="Processing folders"):
        folder_path = os.path.join(root, folder_name)
        tqdm.write(f"Processing folders: {folder_name}")
        
        if os.path.isdir(folder_path) and folder_name in classified_data:
            
            image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]
          
            samples = classified_data[folder_name]

            
            images = [Image.open(image_path) for image_path in image_paths]
            all_images_processed = processor.process_images(images) 
            

            batches_images_processed = [
            {key: tensor[i:i + batch_size] for key, tensor in all_images_processed.items()}
            for i in range(0, len(image_paths), batch_size)
            ]

            # import pdb
            # pdb.set_trace()

            all_embeddings = []
            for batch_images_processed in tqdm(batches_images_processed, desc="Processing images", leave=False):
               
                batch_images_processed = {key: tensor.to(model.device) for key, tensor in batch_images_processed.items()}
                
                with torch.no_grad():
                    image_embeddings = model(**batch_images_processed)

                all_embeddings.append(image_embeddings)

            
            all_embeddings = torch.cat(all_embeddings)
           

            # 逐个处理样本
            for sample in tqdm(samples, desc=f"Processing samples in {folder_name}", leave=False):
                question = sample['question']  # 读取问题
                batch_queries = processor.process_queries([question]).to(model.device) # 处理当前问题

                # 前向传播
                with torch.no_grad():
                    query_embeddings = model(**batch_queries)

                scores = processor.score_single_vector(query_embeddings, all_embeddings)
                # 找到前20个最大得分的索引
                # 获取 scores 的长度
                num_scores = scores.size(1)

                # 动态设置 k 值，如果 scores 的长度小于 20，则使用 scores 的长度作为 k
                k = min(50, num_scores)

                # 获取 top_k_scores 和 top_k_indices
                top_k_scores, top_k_indices = torch.topk(scores, k=k)
                

                # 将前五个最大得分和对应的图片路径添加到样本中
                
                top_k_indices = [idx.item() for idx in top_k_indices[0]]
                top_k_files = [image_paths[idx] for idx in top_k_indices]

                top_page = [
                    int(re.search(r'page_(\d+)', os.path.basename(path)).group(1)) 
                    for path in top_k_files
                ]
                sample['top_pages']=top_page
                with open(out_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(classified_data, json_file, ensure_ascii=False, indent=4)

                



import json
import argparse
import os

# 读取JSON文件
def calculate_ap_at_k(pred, gt, k=10):
    """
    计算 AP@K (Average Precision for top K results)。
    
    参数:
    - pred: 模型预测结果的列表，按相关性分数降序排列。
    - gt: 真实值的列表，包含相关项。
    - k: 只考虑前 k 个预测结果 (默认为 10)。
    
    返回:
    - AP@K 的值。
    """
    k = min(k, len(pred))

    hits = 0
    avg_precision = 0.0
    for rank, item in enumerate(pred[:k], start=1):  # 遍历前 k 个预测结果
        if item in gt:  # 如果预测结果在 GT 中
            hits += 1
            avg_precision += hits / rank  # 累加 Precision@i

    if hits == 0:
        return 0.0  # 如果没有匹配到相关项，AP 为 0
    return avg_precision / len(gt)  # 平均精度归一化到 GT 中的相关项数量 


def F1_score(label_list, result_list):
    TP = 0
    for pred in result_list:
        if pred in label_list:
            TP += 1
    if TP == 0:
        return 0,0,0
    precision = TP / len(result_list)
    # print(precision)
    recall = TP / len(label_list)
    F1 = 2 * precision * recall / (precision+recall)
    if F1>1:
        F1=1
    if precision>1:
        precision=1
    if recall>1:
        recall=1
    return F1, precision, recall

# input_json_file='/home/szy/local/DL_Work_space/colpali_model_infer_test/guji_11_25_2_1_test_data_res/colqwen2-merege-v1.0-dpan-5e-4-warmup30_100k_guji_2_2_data_bs_1_1000_steps.json'
input_json_file=out_json_path
with open(input_json_file, 'r') as f:
    data = json.load(f)

# 创建解析器
parser = argparse.ArgumentParser(description="计算top_k下的检索f1值")

# 添加参数
parser.add_argument('--top_k', type=int, default=10, help="计算分数时选取top_k个结果")

# 解析参数
args = parser.parse_args()

# 遍历每个键值对
F1_score_all=0
precision_score_all=0
recall_score_all=0
count=0
top_k=args.top_k

F1_score_all_sp=0
precision_score_all_sp=0
recall_score_all_sp=0
count_sp=0

F1_score_all_mp=0
precision_score_all_mp=0
recall_score_all_mp=0
count_mp=0


F1_score_all_multi_hop_page=0
precision_score_all_multi_hop_page=0
recall_score_all_multi_hop_page=0
count_multi_hop_page=0


F1_score_all_unanswerable=0
precision_score_all_unanswerable=0
recall_score_all_unanswerable=0
count_unanswerable=0

ap_score_all = 0
ap_score_all_sp = 0
ap_score_all_mp = 0

for key, value in data.items():
    for item in value:
        evidence_pages = item.get('evidence_pages', None)
        top_page = item.get('top_pages', None)
        if evidence_pages:
            evidence_pages=json.loads(evidence_pages)
        # if evidence_pages:
        #     # 处理evidence_pages: 对非0元素减1
        #     evidence_pages = [
        #         (page - 1 if page != 0 else 0) for page in json.loads(evidence_pages)
        #     ]
        if evidence_pages and top_page:
            # import pdb
            # pdb.set_trace()
            F1, precision, recall=F1_score(evidence_pages,top_page[0:top_k])
            ap_score = calculate_ap_at_k(top_page, evidence_pages, k=top_k)
            

            item['F1_score'] = F1
            item['precision'] = precision
            item['recall'] = recall
            item["ap_score"] = ap_score

            # import pdb
            # pdb.set_trace()

            if item["type"]=="单页问答":
                F1_score_all_sp += F1
                precision_score_all_sp += precision
                recall_score_all_sp += recall
                ap_score_all_sp += ap_score
        
                count_sp += 1
                
            elif item["type"]=="跨页问答":
                F1_score_all_mp +=  F1
                precision_score_all_mp += precision
                recall_score_all_mp += recall
                ap_score_all_mp += ap_score
                count_mp += 1
                
            # elif item["type"]=="多跳问答":
            #     F1_score_all_multi_hop_page +=  F1
            #     precision_score_all_multi_hop_page += precision
            #     recall_score_all_multi_hop_page += recall
            #     count_multi_hop_page += 1

            # elif item["type"]=="无法回答":
            #     F1_score_all_unanswerable += F1
            #     precision_score_all_unanswerable += precision
            #     recall_score_all_unanswerable += recall
            #     count_unanswerable += 1

            F1_score_all=F1_score_all + F1
            precision_score_all=precision_score_all + precision
            recall_score_all=recall_score_all + recall
            ap_score_all += ap_score
            count=count+1

F1_score_average=F1_score_all/count*100
precision_score_average=precision_score_all/count*100
recall_score_average=recall_score_all/count*100

F1_score_average_sp = F1_score_all_mp /count_mp*100
precision_score_average_sp = precision_score_all_sp/count_sp*100
recall_score_average_sp = recall_score_all_sp/count_sp*100
map_score_sp = ap_score_all_sp/count_sp*100

F1_score_average_mp = F1_score_all_unanswerable/count_mp*100
precision_score_average_mp = precision_score_all_mp/count_mp*100
recall_score_average_mp = recall_score_all_mp/count_mp*100
map_score_mp = ap_score_all_mp/count_mp*100


# F1_score_average_ua = F1_score_all_unanswerable/count_unanswerable*100
# precision_score_average_ua = precision_score_all_unanswerable/count_unanswerable*100
# recall_score_average_ua = recall_score_all_unanswerable/count_unanswerable*100

map_score_all = ap_score_all/count*100



print(f"result:{out_json_path}")
print(f"Top_k:{top_k}")
# print(f"F1_score_average:{F1_score_average}")
# print(f"precision_score_average:{precision_score_average}")
print(f"recall_score_average_sp:{recall_score_average_sp}")
print(f"map_score_sp:{map_score_sp}")
print(f"recall_score_average_mp:{recall_score_average_mp}")
print(f"map_score_mp:{map_score_mp}")


# output_json_file=input_json_file.split
file_path, file_ext = os.path.splitext(input_json_file)
output_json_file = f"{file_path}_score{file_ext}"

with open(output_json_file, 'w') as f:
    json.dump(data, f, indent=4,ensure_ascii=False)


