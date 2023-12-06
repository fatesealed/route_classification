# coding: UTF-8
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
        self.embedding = nn.Embedding.from_pretrained(
            data_config.embedding_pretrained,
            freeze=False) if data_config.embedding_pretrained is not None else nn.Embedding(data_config.n_vocab,
                                                                                            data_config.embed,
                                                                                            padding_idx=data_config.n_vocab - 1)
        self.embedding_freeze = nn.Embedding.from_pretrained(
            data_config.embedding_pretrained,
            freeze=True) if data_config.embedding_pretrained is not None else nn.Embedding(data_config.n_vocab,
                                                                                           data_config.embed,
                                                                                           padding_idx=data_config.n_vocab - 1)
        self.lstm = nn.GRU(data_config.embed * 2, model_config.hidden_size, model_config.num_layers,
                           bidirectional=True, batch_first=True, dropout=model_config.dropout)
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_ size * 2))
        self.w = nn.Parameter(torch.rand(model_config.hidden_size * 2))
        self.fc = nn.Linear(model_config.hidden_size * 2, model_config.hidden_size)
        self.fc2 = nn.Linear(model_config.hidden_size, data_config.num_classes)
        self.mutilatte = nn.MultiheadAttention(embed_dim=model_config.hidden_size * 2, num_heads=8, batch_first=True)

    def forward(self, x):
        emb_x = torch.cat([self.embedding(x), self.embedding_freeze(x)],
                          dim=2)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb_x)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        # M = F.relu(H)  # [128, 32, 256]
        # alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        # out = H * alpha  # [128, 32, 256]

        # 👆原始实现 👇pytorch版

        out, _ = self.mutilatte(H, H, H)
        out = torch.sum(out, dim=1)  # 求和
        out = F.relu(out)
        out = self.fc(out)  # [128, 64]
        out = self.fc2(out)
        return out
