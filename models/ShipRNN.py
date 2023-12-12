# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig(object):
    """配置参数"""

    def __init__(self, notes=''):
        self.model_name = 'ShipRNN'
        self.save_path = f'./result/{self.model_name}_{notes}.ckpt'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name

        self.dropout = 0.5  # 随机失活
        self.hidden_size = 256  # lstm隐藏层
        self.num_layers = 2  # lstm层数


class Model(nn.Module):
    def __init__(self, model_config, data_config):
        super(Model, self).__init__()
        print(data_config.n_vocab, data_config.embed)
        self.embedding = nn.Embedding.from_pretrained(
            data_config.embedding_pretrained,
            freeze=False) if data_config.embedding_pretrained is not None else nn.Embedding(data_config.n_vocab,
                                                                                            data_config.embed,
                                                                                            padding_idx=data_config.n_vocab - 1)
        self.lstm = nn.LSTM(data_config.embed, model_config.hidden_size, model_config.num_layers,
                            bidirectional=True, batch_first=True, dropout=model_config.dropout)
        self.bn = nn.BatchNorm1d(model_config.hidden_size * 2)
        self.avg_pool = nn.AvgPool1d(data_config.pad_size)
        self.mutilatte = nn.MultiheadAttention(embed_dim=model_config.hidden_size * 2 + data_config.embed, num_heads=6,
                                               batch_first=True)
        self.fc = nn.Linear(model_config.hidden_size * 2 + data_config.embed, data_config.num_classes)

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]
        out, _ = self.lstm(embed)  # 左右双向
        out1 = torch.cat((embed, out), 2)
        out1 = F.gelu(out1)
        out2, _ = self.mutilatte(out1, out1, out1)
        out2 = out2.permute(0, 2, 1)
        out2 = self.avg_pool(out2).squeeze()
        out2 = F.gelu(out2)
        out2 = self.fc(out2)  # 句子最后时刻的 hidden state
        return out2
