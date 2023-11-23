# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig(object):
    """配置参数"""

    def __init__(self, notes=''):
        self.model_name = 'TextRNN_Att'
        self.save_path = f'./result/{self.model_name}_{notes}.ckpt'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name

        self.dropout = 0.5  # 随机失活
        self.hidden_size = 256  # lstm隐藏层
        self.num_layers = 2  # lstm层数


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, model_config, data_config):
        super(Model, self).__init__()
        if data_config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(data_config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(data_config.n_vocab, data_config.embed, padding_idx=data_config.n_vocab - 1)
        self.lstm = nn.LSTM(data_config.embed, model_config.hidden_size, model_config.num_layers,
                            bidirectional=True, batch_first=True, dropout=model_config.dropout)
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.rand(model_config.hidden_size * 2))
        self.fc = nn.Linear(model_config.hidden_size * 2, data_config.num_classes)

    def forward(self, x):
        emb_x = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb_x)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
        M = F.tanh(H)  # [128, 32, 256]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, dim=1)  # 求和
        out = F.relu(out)
        out = self.fc(out)  # [128, 64]
        return out
