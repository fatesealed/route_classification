# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'FastText'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name

        self.dropout = 0.5  # 随机失活
        self.hidden_size = 256  # 隐藏层大小
        self.n_gram_vocab = 250499  # ngram 词表大小


'''Bag of Tricks for Efficient Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):

        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
