# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding, notes):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/train_dataset.csv'
        self.val_path = dataset + '/val_dataset.csv'
        self.test_path = dataset + '/test_dataset.csv'
        self.class_list = [x.strip() for x in
                           open(dataset + '/pre_data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/pre_data/vocab.pkl'  # 词表
        self.save_path = f'./result/{self.model_name}_{notes}.ckpt'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/pre_data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.is_random = "random" if embedding == "random" else "not_random"

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 8000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 100  # epoch数
        self.batch_size = 512  # mini-batch大小
        self.pad_size = 30  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 20  # 字向量维度
        self.filter_sizes = (2, 3, 4, 5)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


def conv_and_pool(x, conv):
    x = f.relu(conv(x))
    x = x.squeeze(3)
    x = f.max_pool1d(x, x.size(2)).squeeze(2)
    return x


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding_1 = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            self.embedding_2 = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding_1 = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
            self.embedding_2 = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed * 2)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(config.num_filters * len(config.filter_sizes),
                      config.num_filters * len(config.filter_sizes) // 2),
            nn.Linear(config.num_filters * len(config.filter_sizes) // 2,
                      config.num_filters * len(config.filter_sizes) // 4),
            nn.Linear(config.num_filters * len(config.filter_sizes) // 4, config.num_classes))

    def forward(self, x):
        out = torch.cat((self.embedding_1(x), self.embedding_2(x)), dim=2)
        out = out.unsqueeze(1)  # 插入维度 进行卷积运算
        out = torch.cat([conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_layers(out)
        return out
