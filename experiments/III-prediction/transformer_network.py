import torch
import torch.nn as nn
import math

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=103):
        super(LearnedPositionalEncoding, self).__init__()

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
        self.d_model = d_model

        # 创建相对位置编码的模块
        self.positional_encoding = LearnedPositionalEncoding(d_model)

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

    def forward(self, src, mask):
        # 缩放输入，以平衡Transformer中的梯度流
        src = src * math.sqrt(self.d_model)

        # 添加相对位置编码到输入张量中
        src = self.positional_encoding(src)

        # 通过Transformer编码器处理输入序列
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=mask)

        # 全局平均池化，将每个序列的所有位置的特征平均为一个单一的向量
        # pooled_output = torch.mean(encoder_output, dim=1)

        # 使用线性层进行分类，并应用对数软最大化
        # logits = self.fc(encoder_output)
        logits = self.fc(encoder_output[:, -1, :])

        return logits
