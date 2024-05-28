import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=103):
        super(PositionalEncoding, self).__init__()

        # 创建一个形状为 (max_len, d_model) 的零矩阵
        pe = torch.zeros(max_len, d_model)

        # 创建一个表示位置的序列，然后将其形状调整为 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算用于位置编码的除数项，然后将其形状调整为 (d_model/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码增加一个维度，并注册为模型的缓冲区
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, max_len=103):
        # 在输入张量 x 上加上位置编码，仅使用与输入序列长度相对应的位置编码
        return x + self.pe[:, :max_len].detach()

class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, d_model, max_len=103):
        super(LearnedPositionalEmbeddings, self).__init__()

        # 创建可学习的位置嵌入参数
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe

class NoAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(NoAttentionEncoderLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout1(src2)
        src2 = self.linear2(src2)
        src2 = self.dropout2(src2)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, num_classes):
        super(TransformerModel, self).__init__()

        # 定义模型的嵌入维度
        self.embedding_dim = d_model

        # 创建相对位置编码的模块
        self.positional_encoding = LearnedPositionalEmbeddings(d_model)

        # 创建Transformer编码器，其中包括多个Transformer编码层
        # encoder_layers = NoAttentionEncoderLayer(d_model, dim_feedforward, dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                       dropout=dropout, batch_first=True),
            num_layers=num_layers
        )

        # 创建线性层，用于将Transformer的输出映射到类别分数
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask):
        # 缩放输入，以平衡Transformer中的梯度流
        x = x * math.sqrt(self.embedding_dim)

        # 添加相对位置编码到输入张量中
        x = self.positional_encoding(x)

        # 通过Transformer编码器处理输入序列
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # 全局平均池化，将每个序列的所有位置的特征平均为一个单一的向量
        # x = torch.mean(x, dim=2)  # Global average pooling

        # 使用线性层进行分类，并应用对数软最大化
        x = self.fc(x[:, -1, :])

        return x
