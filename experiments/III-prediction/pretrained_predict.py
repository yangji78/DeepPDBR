import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from pretrained_networks import Bert
import os

root_path = 'E:/projects/docker-configuration-23/dockerfile/'
model_save_path = root_path + 'experiments/model/bert_best.pth'

learning_rate = 0.0005
epochs = 10
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self):
        self.features_dir = 'E:/projects/dataset_docker/features_bert'
        self.labels_dir = 'E:/projects/dataset_docker/labels'
        self.masks_dir = 'E:/projects/dataset_docker/masks_bert'

        self.features_file_list = os.listdir(self.features_dir)
        self.labels_file_list = os.listdir(self.labels_dir)
        self.masks_file_list = os.listdir(self.masks_dir)

    def __len__(self):
        return len(self.features_file_list)

    def __getitem__(self, idx):
        feature_filename = os.path.join(self.features_dir, self.features_file_list[idx])
        label_filename = os.path.join(self.labels_dir, self.labels_file_list[idx])
        mask_filename = os.path.join(self.masks_dir, self.masks_file_list[idx])

        # 加载数据样本、标签和数据向量长度
        features = np.load(feature_filename)
        labels = np.load(label_filename)
        masks = np.load(mask_filename)

        # 转换为PyTorch张量，移到GPU
        features = torch.tensor(features, dtype=torch.long).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        masks = torch.tensor(masks, dtype=torch.float32).to(device)

        return features, labels, masks

# 定义评估函数
def evaluate(y_true, y_out, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_out = y_out.cpu().numpy()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_out)

    return precision, recall, f1, auc

# 初始化模型和优化器
model = Bert()
model.to(device)

# 初始化数据集
dataset = MyDataset()

# 划分数据集为训练集、验证集和测试集
total_size = len(dataset)
# train_size = int(0.8 * total_size)
# val_size = int(0.1 * total_size)
# test_size = total_size - train_size - val_size
train_size = 300
test_size = int(0.1 * total_size)
val_size = int(0.1 * total_size)
nouse_size = total_size - train_size - test_size - val_size
train_dataset, val_dataset, test_dataset, nouse_dataset = random_split(dataset, [train_size, val_size, test_size, nouse_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数，并设置类别权重
pos_weight = torch.tensor([0.45]).to(device)  # 替换为你的权重值
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 二元分类任务使用BCEWithLogitsLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

val_best_loss = float('inf')
last_improve = 0  # 记录上次验证集loss下降的batch数
# 训练模型
for epoch in range(epochs):
    model.train()
    train_loss = []
    for batch_features, batch_labels, batch_masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        optimizer.zero_grad()
        train_outputs = model(batch_features, batch_masks)
        loss = criterion(train_outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    train_loss = np.mean(train_loss, axis=0)
    print(f'Epoch {epoch + 1}, Train loss: {train_loss:.4f}')

    # 在验证集上评估模型
    model.eval()
    val_results = []
    val_loss = []
    with torch.no_grad():
        for batch_features, batch_labels, batch_masks in tqdm(val_loader, desc='Validation'):
            val_outputs = model(batch_features, batch_masks)
            val_predictions = (torch.sigmoid(val_outputs) > 0.5).float()
            try:
                precision, recall, f1, auc = evaluate(batch_labels, val_outputs, val_predictions)
            except ValueError:
                continue
            val_results.append([precision, recall, f1, auc])

            loss = criterion(val_outputs, batch_labels)
            val_loss.append(loss.item())

    val_loss = np.mean(val_loss, axis=0)
    print(f'Epoch {epoch + 1}, Val loss: {val_loss:.4f}')
    if val_loss < val_best_loss:
        val_best_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        last_improve = epoch

    val_results = np.mean(val_results, axis=0)  # 计算验证集的平均结果
    print("Validation Results:")
    print(f"Precision: {val_results[0]:.4f}")
    print(f"Recall: {val_results[1]:.4f}")
    print(f"F1 Score: {val_results[2]:.4f}")
    print(f"AUC: {val_results[3]:.4f}")

    scheduler.step()  # 学习率衰减

    if epoch - last_improve > 1:
        # 验证集loss超过1 epoch没下降，结束训练
        print("No optimization for a long time, auto-stopping...")
        break

# 在测试集上评估模型
model.load_state_dict(torch.load(model_save_path))
model.eval()
test_results = []
test_loss = []
with torch.no_grad():
    for batch_features, batch_labels, batch_masks in tqdm(test_loader, desc='Testing'):
        test_outputs = model(batch_features, batch_masks)
        test_predictions = (torch.sigmoid(test_outputs) > 0.5).float()
        try:
            precision, recall, f1, auc = evaluate(batch_labels, test_outputs, test_predictions)
        except ValueError:
            continue
        test_results.append([precision, recall, f1, auc])

        loss = criterion(test_outputs, batch_labels)
        test_loss.append(loss.item())

test_loss = np.mean(test_loss, axis=0)
print(f'Test loss: {test_loss:.4f}')

test_results = np.mean(test_results, axis=0)  # 计算测试集的平均结果
print("Test Results:")
print(f"Precision: {test_results[0]:.4f}")
print(f"Recall: {test_results[1]:.4f}")
print(f"F1 Score: {test_results[2]:.4f}")
print(f"AUC: {test_results[3]:.4f}")

