import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformer_network import TransformerModel
import os
from sklearn.model_selection import KFold
from datetime import datetime

best_model_save_path = 'E:/projects/docker-configuration-23/dockerfile/experiments/model/transformer_best.pth'

# 超参数
d_model = 100
nhead = 4
num_layers = 3
dim_feedforward = 1024
dropout = 0.1
num_classes = 1
learning_rate = 0.0005
epochs = 100
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self):
        self.features_dir = 'E:/projects/dataset_docker/features'
        self.labels_dir = 'E:/projects/dataset_docker/labels'
        self.masks_dir = 'E:/projects/dataset_docker/masks'
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
        features = torch.tensor(features, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        masks = torch.tensor(masks, dtype=torch.float32).to(device)

        return features, labels, masks

# 定义评估函数
def evaluate(y_true, y_prob, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_prob = y_prob.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    return accuracy, precision, recall, f1, auc

# 初始化数据集
dataset = MyDataset()

# 定义交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化存储结果的列表
test_results = []
test_losses = []

time1 = datetime.now()

# 开始交叉验证
for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/{kf.get_n_splits()}')

    # 划分数据集为训练集和测试集
    train_dataset_fold = Subset(dataset, train_index)
    test_dataset_fold = Subset(dataset, test_index)

    # 再将训练集划分为训练集和验证集
    train_size = int(0.9 * len(train_dataset_fold))
    val_size = len(train_dataset_fold) - train_size
    train_subset_fold, val_subset_fold = random_split(train_dataset_fold, [train_size, val_size])

    train_loader = DataLoader(train_subset_fold, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset_fold, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset_fold, batch_size=batch_size, shuffle=False)

    # 初始化模型和优化器
    model = TransformerModel(d_model, nhead, num_layers, dim_feedforward, dropout, num_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pos_weight = torch.tensor([0.45]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # 训练和验证
    val_best_loss = float('inf')
    last_improve = 0
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

        train_loss = np.mean(train_loss)
        print(f'Epoch {epoch + 1}, Train loss: {train_loss:.4f}')

        # 在验证集上评估
        model.eval()
        val_results = []
        val_loss = []
        with torch.no_grad():
            for batch_features, batch_labels, batch_masks in tqdm(val_loader, desc='Validation'):
                val_outputs = model(batch_features, batch_masks)
                val_prob = torch.sigmoid(val_outputs)
                val_predictions = (val_prob > 0.5).float()
                try:
                    accuracy, precision, recall, f1, auc = evaluate(batch_labels, val_prob, val_predictions)
                except ValueError:
                    continue
                val_results.append([accuracy, precision, recall, f1, auc])

                loss = criterion(val_outputs, batch_labels)
                val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        print(f'Epoch {epoch + 1}, Val loss: {val_loss:.4f}')

        if val_loss < val_best_loss:
            val_best_loss = val_loss
            torch.save(model.state_dict(), best_model_save_path)
            last_improve = epoch

        val_results = np.mean(val_results, axis=0)
        print("Validation Results:")
        print(f"Accuracy: {val_results[0]:.4f}")
        print(f"Precision: {val_results[1]:.4f}")
        print(f"Recall: {val_results[2]:.4f}")
        print(f"F1 Score: {val_results[3]:.4f}")
        print(f"AUC: {val_results[4]:.4f}")

        scheduler.step()  # 学习率衰减

        if epoch - last_improve > 1:
            # 验证集loss超过1 epoch没下降，结束训练
            print("No optimization for a long time, early-stopping...")
            break

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(best_model_save_path))
    model.eval()
    test_results_fold = []
    test_loss_fold = []
    with torch.no_grad():
        for batch_features, batch_labels, batch_masks in tqdm(test_loader, desc='Testing'):
            test_outputs = model(batch_features, batch_masks)
            test_prob = torch.sigmoid(test_outputs)
            test_predictions = (test_prob > 0.5).float()
            try:
                accuracy, precision, recall, f1, auc = evaluate(batch_labels, test_prob, test_predictions)
            except ValueError:
                continue
            test_results_fold.append([accuracy, precision, recall, f1, auc])

            loss = criterion(test_outputs, batch_labels)
            test_loss_fold.append(loss.item())

    test_loss_fold = np.mean(test_loss_fold)
    print(f'Fold {fold + 1}, Test loss: {test_loss_fold:.4f}')

    test_results_fold = np.mean(test_results_fold, axis=0)
    test_results.append(test_results_fold)
    test_losses.append(test_loss_fold)

# 计算所有折的平均结果
avg_test_results = np.mean(test_results, axis=0)
avg_test_loss = np.mean(test_losses)

print("\nAverage Test Results across all folds:")
print(f"Accuracy: {avg_test_results[0]:.4f}")
print(f"Precision: {avg_test_results[1]:.4f}")
print(f"Recall: {avg_test_results[2]:.4f}")
print(f"F1 Score: {avg_test_results[3]:.4f}")
print(f"AUC: {avg_test_results[4]:.4f}")

print(f"\nAverage Test Loss across all folds: {avg_test_loss:.4f}")

time2 = datetime.now()
time_difference = time2 - time1
seconds = time_difference.seconds
hours = seconds // 3600
minutes = (seconds % 3600) // 60
seconds = seconds % 60
print(f"时间差: {hours}小时 {minutes}分钟 {seconds}秒")