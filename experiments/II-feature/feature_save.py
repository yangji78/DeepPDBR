import pandas as pd
import json
from gensim.models import Word2Vec
import numpy as np
import os

# 加载Dockerfile AST数据和标签
root_path = 'E:/projects/docker-configuration-23/dockerfile/data/'
jsonl_file_path = root_path + 'dataset.jsonl'  # 替换为实际的文件路径
labels = pd.read_csv(root_path + 'labels_r.csv')

# 加载Word2Vec模型
word2vec_model_path = root_path + 'word2vec/words.model'  # 替换为你的Word2Vec模型路径
word2vec_model = Word2Vec.load(word2vec_model_path)

# 创建保存特征数据的目录
features_save_dir = 'E:/projects/dataset_docker/features'
labels_save_dir = 'E:/projects/dataset_docker/labels'
masks_save_dir = 'E:/projects/dataset_docker/masks'
lengths_save_dir = 'E:/projects/dataset_docker/lengths'

# 递归函数来提取"type"的值并生成句子的词嵌入向量
def extract_type_values(json_obj):
    sentence = ""

    def append_to_sentence(value):
        nonlocal sentence
        if sentence and sentence[-1] != '_':
            sentence += "_"
        sentence += value

    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key == "type" and value != 'UNKNOWN':
                append_to_sentence(value)
            elif isinstance(value, (dict, list)):
                append_to_sentence(extract_type_values(value))
    elif isinstance(json_obj, list):
        for item in json_obj:
            append_to_sentence(extract_type_values(item))

    # 删除末尾的下划线
    if sentence and sentence[-1] == '_':
        sentence = sentence[:-1]

    return sentence

def get_ast_vec(json_obj):
    sentence_vectors = []

    list = json_obj.get("children")
    for item in list:
        value = extract_type_values(item)
        if value in word2vec_model.wv:
            sentence_vectors.append(word2vec_model.wv[value])

    return sentence_vectors

# 逐行读取JSONL文件并提取特征和标签
features = []
max_feature_length = 0
json_count = 0
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        # 解析JSON对象
        json_obj = json.loads(line)

        # 调用递归函数来提取"type"的值并生成句子的词嵌入向量
        sentence_vectors = get_ast_vec(json_obj)

        if sentence_vectors:
            features.append(sentence_vectors)
            max_feature_length = max(max_feature_length, len(sentence_vectors))

        json_count += 1

# 打印sentences的长度和最大sentence的长度
print(f"Total sentences: {len(features)}") # 37727
print(f"Max sentence length: {max_feature_length}") # 3656 103

labels = np.array(labels)
sequence_lengths = np.array([len(seq) for seq in features])

# 找到所有批次中的最长序列长度
max_sequence_length = max(len(seq) for seq in features)

for i in range(0, len(features)):
    seq = features[i]
    padding_length = max_sequence_length - len(seq)
    padded_seq = np.vstack([seq, np.zeros((padding_length, 100))])
    # np.save(os.path.join(features_save_dir, f'padded_features_{i + 1}.npy'), padded_seq)

    max_len = 103

    labels_batch = labels[i]
    sequence_lengths_batch = sequence_lengths[i]
    masks_batch = np.array([0] * sequence_lengths_batch + [1] * (max_len - sequence_lengths_batch))
    lengths_batch = len(seq)

    # 保存每个批次的特征数据
    # np.save(os.path.join(labels_save_dir, f'labels_{i + 1}.npy'), labels_batch)
    # np.save(os.path.join(masks_save_dir, f'masks_{i + 1}.npy'), masks_batch)
    np.save(os.path.join(lengths_save_dir, f'lengths_{i + 1}.npy'), lengths_batch)

print(f"处理完毕，处理了{json_count}个")
print(f'最大特征向量长度为 {max_feature_length}')
