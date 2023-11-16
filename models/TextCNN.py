# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as f


class Config(object):
    """配置参数"""

    def __init__(self, notes=''):
        self.model_name = 'TextCNN'
        self.save_path = f'./result/{self.model_name}_{notes}.ckpt'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 10000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 100  # epoch数
        self.batch_size = 512  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率
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
        if data_config.embedding_pretrained is not None:
            self.embedding_1 = nn.Embedding.from_pretrained(data_config.embedding_pretrained, freeze=False)
            self.embedding_2 = nn.Embedding.from_pretrained(data_config.embedding_pretrained, freeze=False)
        else:
            self.embedding_1 = nn.Embedding(data_config.n_vocab, data_config.embed,
                                            padding_idx=data_config.n_vocab - 1)
            self.embedding_2 = nn.Embedding(data_config.n_vocab, data_config.embed,
                                            padding_idx=data_config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, model_config.num_filters, (k, data_config.embed * 2)) for k in model_config.filter_sizes])
        self.dropout = nn.Dropout(model_config.dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(model_config.num_filters * len(model_config.filter_sizes),
                      model_config.num_filters * len(model_config.filter_sizes) // 2),
            nn.Linear(model_config.num_filters * len(model_config.filter_sizes) // 2,
                      model_config.num_filters * len(model_config.filter_sizes) // 4),
            nn.Linear(model_config.num_filters * len(model_config.filter_sizes) // 4, data_config.num_classes))

    def forward(self, x):
        out = torch.cat((self.embedding_1(x), self.embedding_2(x)), dim=2)
        out = out.unsqueeze(1)  # 插入维度 进行卷积运算
        out = torch.cat([conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_layers(out)
        return out
