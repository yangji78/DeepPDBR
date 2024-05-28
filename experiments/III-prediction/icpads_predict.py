import json
import random

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from datetime import datetime

# 打开JSONL文件并读取数据
root_path = 'E:/projects/docker-configuration-23/dockerfile/data/'
jsonl_file_path = root_path + 'dataset.jsonl'  # JSONL文件路径
# 加载标签数据
labels = pd.read_csv(root_path + 'labels_r.csv')

mapping = {
    "DOCKER-ADD": 1,
    "DOCKER-EXPOSE": 2,
    "DOCKER-SHELL": 3,
    "DOCKER-ENTRYPOINT": 4,
    "DOCKER-ENV": 5,
    "DOCKER-FROM": 6,
    "DOCKER-RUN": 7,
    "DOCKER-CMD": 8,
    "DOCKER-VOLUME": 9,
    "DOCKER-ARG": 10,
    "DOCKER-WORKDIR": 11,
    "DOCKER-COPY": 12,
    "DOCKER-USER": 13,
    "DOCKER-ADD-TARGET": 14,
    "DOCKER-ADD-SOURCE": 15,
    "DOCKER-PORT": 16,
    "DOCKER-SHELL-EXECUTABLE": 17,
    "DOCKER-ENTRYPOINT-EXECUTABLE": 18,
    "DOCKER-IMAGE-NAME": 19,
    "DOCKER-IMAGE-TAG": 20,
    "DOCKER-IMAGE-REPO": 21,
    "BASH-SCRIPT": 22,
    "DOCKER-CMD-ARG": 23,
    "DOCKER-COPY-TARGET": 24,
    "DOCKER-COPY-SOURCE": 25,
    "DOCKER-NAME": 26,
    "DOCKER-LITERAL": 27,
    "DOCKER-PATH": 28
}

# 递归函数来提取"type"的值并生成句子的词汇列表
def extract_type_values(json_obj):
    sentence = []
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key == "type":
                value0 = value.split(":")[0] if ":" in value else value
                if value0 in mapping:
                    sentence.append(mapping[value0])
            elif isinstance(value, (dict, list)):
                sentence.extend(extract_type_values(value))
    elif isinstance(json_obj, list):
        for item in json_obj:
            sentence.extend(extract_type_values(item))
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
        sentence = extract_type_values(json_obj)

        if sentence:
            # 在sentence后面添加0，直到长度达到一致
            while len(sentence) < 338:
                sentence.append(0)
            sentences.append(sentence)
            max_sentence_length = max(max_sentence_length, len(sentence))
            json_count += 1

# 打印sentences的长度和最大sentence的长度
print(f"Total sentences: {len(sentences)}") # 37727
print(f"Max sentence length: {max_sentence_length}") # 338

# 定义评估函数
def etestuate(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return precision, recall, f1, auc

# Combine sentences and labels
df = pd.DataFrame({'sentence': sentences, 'label': labels['label']})

# Define features (X) and labels (y)
X = df['sentence']
y = df['label']

# Initialize classifiers
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', max_iter=1000)
lr = LogisticRegression(penalty='l2')
dt = DecisionTreeClassifier(splitter='best')
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
nb = BernoulliNB()

classifiers = [knn, svm, lr, dt, rf, nb]

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define evaluation function
def evaluate_classifier(classifier, X, y):
    fold = 1
    results = []
    print(f"{classifier.__class__.__name__}")
    for train_index, test_index in cv.split(X, y):
        print(f"Fold: {fold}")
        fold += 1

        # Split the data into train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train = X_train.to_list()
        X_test = X_test.to_list()
        y_train = y_train.to_list()
        y_test = y_test.to_list()

        selected_indices = random.sample(range(len(X_train)), 500)

        # 根据随机选择的索引提取对应的训练数据和标签
        X_train_sampled = [X_train[i] for i in selected_indices]
        y_train_sampled = [y_train[i] for i in selected_indices]

        # Fit the classifier on the training set
        classifier.fit(X_train_sampled, y_train_sampled)

        # Predict on the test set
        y_test_pred = classifier.predict(X_test)

        # Calculate additional evaluation metrics on the test set
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_pred)
        results.append([precision, recall, f1, auc])

    results = np.mean(results, axis=0)  # 计算测试集的平均结果

    # Print the values for each metric
    print(f"Precision: {results[0]:.4f}")
    print(f"Recall: {results[1]:.4f}")
    print(f"F1 Score: {results[2]:.4f}")
    print(f"AUC: {results[3]:.4f}")

# Evaluate each classifier
for clf in classifiers:
    time1 = datetime.now()
    evaluate_classifier(clf, X, y)
    time2 = datetime.now()
    time_difference = time2 - time1
    seconds = time_difference.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    print(f"时间差: {hours}小时 {minutes}分钟 {seconds}秒")
