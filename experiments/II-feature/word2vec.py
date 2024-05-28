import json
from gensim.models import Word2Vec

# 打开JSONL文件并读取数据
root_path = 'E:/projects/docker-configuration-23/dockerfile/data/'
jsonl_file_path = root_path + 'dataset.jsonl'  # JSONL文件路径
word2vec_model_path = root_path + 'word2vec/words.model'  # 保存Word2Vec模型的路径

# 递归函数来提取"type"的值并生成句子的词汇列表
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

def get_ast(json_obj):
    sentence = []

    list = json_obj.get("children")
    for item in list:
        sentence.append(extract_type_values(item))

    return sentence

# 逐行读取JSONL文件并构建训练Word2Vec的语料库
sentences = []
max_sentence_length = 0  # 用于记录最大句子长度
json_count = 0  # 用于记录处理的JSON对象数量

with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        # 解析JSON对象
        json_obj = json.loads(line)

        # 调用递归函数来提取"type"的值并生成句子的词汇列表
        sentence = get_ast(json_obj)

        if sentence:
            sentences.append(sentence)
            max_sentence_length = max(max_sentence_length, len(sentence))

        json_count += 1

# 打印sentences的长度和最大sentence的长度
print(f"Total sentences: {len(sentences)}") # 37727
print(f"Max sentence length: {max_sentence_length}") # 3656 103

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=10, min_count=1, sg=1, epochs=10, workers=8)

# 打印词汇表大小
vocabulary_size = len(model.wv.key_to_index)
print(f"Vocabulary size: {vocabulary_size}") # 16392 23836

# 打印嵌入维度
embedding_dimension = model.vector_size
print(f"Embedding dimension: {embedding_dimension}") # 100

# 保存训练好的模型到文件
model.save(word2vec_model_path)

print("Word2Vec模型已训练并保存。")
