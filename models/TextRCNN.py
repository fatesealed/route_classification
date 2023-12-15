# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig(object):
    """配置参数"""

    def __init__(self, freeze, notes=''):
        self.freeze = freeze
        self.model_name = 'TextRCNN'
        self.save_path = f'./result/{self.model_name}_{notes}.ckpt'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name

        self.dropout = 0.5  # 随机失活
        self.hidden_size = 256  # lstm隐藏层
        self.num_layers = 2  # lstm层数


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, model_config, data_config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            data_config.embedding_pretrained,
            freeze=model_config.freeze) if data_config.embedding_pretrained is not None else nn.Embedding(
            data_config.n_vocab,
            data_config.embed,
            padding_idx=data_config.n_vocab - 1)
        self.lstm = nn.LSTM(data_config.embed, model_config.hidden_size, model_config.num_layers,
                            bidirectional=True, batch_first=True, dropout=model_config.dropout)
        self.fc = nn.Linear(model_config.hidden_size * 2 + data_config.embed, data_config.num_classes)
        self.maxpool = nn.MaxPool1d(data_config.pad_size)

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        # _, top_indices = torch.topk(out, k=1, dim=2)
        # out = torch.gather(out, 1, top_indices).squeeze()
        out = self.fc(out)
        return out
