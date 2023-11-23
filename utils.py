# coding: UTF-8
import os
import pickle as pkl
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class DataConfig:
    def __init__(self, dim, embedding):
        dataset = 'ship_data'
        self.train_path = os.path.join(dataset, 'train_dataset.csv')
        self.val_path = os.path.join(dataset, 'val_dataset.csv')
        self.test_path = os.path.join(dataset, 'test_dataset.csv')
        self.class_list = [x.strip() for x in
                           open(os.path.join(dataset, 'pre_data', 'class.txt'), encoding='utf-8').readlines()]
        self.vocab_path = os.path.join(dataset, 'pre_data', 'vocab.pkl')
        embedding_path = os.path.join(dataset, 'pre_data', f'{dim}_{embedding}.npz')
        self.embedding_pretrained = torch.tensor(
            np.load(embedding_path)["embeddings"].astype('float32')) if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_random = "random" if embedding == "random" else "not_random"
        self.num_classes = len(self.class_list)  # 类别数
        # 设置字向量维度为预训练维度或默认值
        self.embed = (
            self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 100)
        self.pad_size = 30  # 每句话处理成的长度(短填长切)
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.require_improvement = 10000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 100  # epoch数
        self.batch_size = 1024  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率


# 自定义数据集类，需要实现__len__和__getitem__方法
class CustomDataset(Dataset):
    def __init__(self, config, data_class):
        tokenizer = lambda x: x.split('|')  # word-level
        if data_class == 'train':
            path = config.train_path
        elif data_class == 'val':
            path = config.val_path
        elif data_class == 'test':
            path = config.test_path
        vocab = pkl.load(open(config.vocab_path, 'rb'))  # 打开词表
        class_int_dict = {item: i for i, item in enumerate(config.class_list)}
        df = pd.read_csv(path, usecols=['path', 'cluster'])  # 读取csv
        contents = []
        for index, row in df.iterrows():
            content, label = row['path'], row['cluster']
            token = tokenizer(content)
            seq_len = len(token)
            if config.pad_size:  # 统一长度
                if seq_len < config.pad_size:
                    token.extend([PAD] * (config.pad_size - len(token)))
                else:
                    token = token[:config.pad_size]
                    seq_len = config.pad_size
            words_line = []
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, class_int_dict[label], seq_len))
        self.data = contents

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.LongTensor(self.data[i][0]), self.data[i][1], self.data[i][2]


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
