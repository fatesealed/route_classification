# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as f


class ModelConfig(object):
    """配置参数"""

    def __init__(self, notes=''):
        self.model_name = 'TextCNN'
        self.save_path = f'./result/{self.model_name}_{notes}.pth'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name

        self.dropout = 0.5  # 随机失活
        self.filter_sizes = (2, 3, 4, 5)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


def conv_and_pool(x, conv):
    x = f.relu(conv(x))
    x = x.squeeze(3)
    x = f.max_pool1d(x, x.size(2)).squeeze(2)
    return x


class Model(nn.Module):
    def __init__(self, model_config, data_config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            data_config.embedding_pretrained,
            freeze=False) if data_config.embedding_pretrained is not None else nn.Embedding(data_config.n_vocab,
                                                                                            data_config.embed,
                                                                                            padding_idx=data_config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, model_config.num_filters, (k, data_config.embed)) for k in model_config.filter_sizes])
        self.dropout = nn.Dropout(model_config.dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(model_config.num_filters * len(model_config.filter_sizes), data_config.num_classes))

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)  # 插入维度 进行卷积运算
        out = torch.cat([conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_layers(out)
        return out
