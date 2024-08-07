# coding: UTF-8
import os
import pickle as pkl
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class DataConfig:
    def __init__(self, embedding, dim, class_type):
        self.dim = dim
        dataset = 'ship_data'
        self.class_type = class_type
        self.train_path = os.path.join(dataset, 'img_train_dataset.csv')
        self.val_path = os.path.join(dataset, 'img_val_dataset.csv')
        self.test_path = os.path.join(dataset, 'img_test_dataset.csv')
        self.class_list = [x.strip() for x in
                           open(os.path.join(dataset, 'pre_data', f'{class_type}_class.txt'),
                                encoding='utf-8').readlines()]
        self.vocab_path = os.path.join(dataset, 'pre_data', 'vocab.pkl')
        embedding_path = os.path.join(dataset, 'pre_data', f'{self.dim}d_{embedding}.npz')
        self.embedding_pretrained = torch.tensor(
            np.load(embedding_path)["embeddings"].astype('float32')) if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_random = "random" if embedding == "random" else "not_random"
        self.num_classes = len(self.class_list)  # 类别数
        # 设置字向量维度为预训练维度或默认值
        self.embed = (
            self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else self.dim)
        self.pad_size = 30  # 每句话处理成的长度(短填长切)
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.require_improvement = 2000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 100  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率


class BertDataConfig:
    def __init__(self, ):
        dataset = 'bert_data'
        self.train_path = os.path.join(dataset, 'train_dataset.csv')
        self.val_path = os.path.join(dataset, 'val_dataset.csv')
        self.test_path = os.path.join(dataset, 'test_dataset.csv')
        self.class_list = [x.strip() for x in
                           open(os.path.join(dataset, 'pre_data', 'class.txt'), encoding='utf-8').readlines()]
        self.vocab_path = os.path.join(dataset, 'pre_data', 'vocab.pkl')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.class_list)  # 类别数
        self.pad_size = 30  # 每句话处理成的长度(短填长切)
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.require_improvement = 10000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 100  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率


# 自定义数据集类，需要实现__len__和__getitem__方法
class CustomDataset(Dataset):
    def __init__(self, config, data_class):
        global path
        tokenizer = lambda x: x.split('|')  # word-level
        if data_class == 'train':
            path = config.train_path
        elif data_class == 'val':
            path = config.val_path
        elif data_class == 'test':
            path = config.test_path
        vocab = pkl.load(open(config.vocab_path, 'rb'))  # 打开词表
        class_int_dict = {item: i for i, item in enumerate(config.class_list)}
        df = pd.read_csv(path)  # 读取csv
        contents = []
        transform = transforms.ToTensor()
        class_type = config.class_type
        for index, row in df.iterrows():
            content, label = row['path'], row[class_type]
            mmsi = row['mmsi']
            pic = Image.open(f'./ship_data/pic_data/mmsi_{mmsi}.png')
            pic = transform(pic)
            token = tokenizer(content)
            seq_len = len(token)
            if config.pad_size:  # 统一长度
                if seq_len < config.pad_size:
                    token.extend([PAD] * (config.pad_size - len(token)))
                else:
                    token = token[:config.pad_size]
            words_line = []
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, class_int_dict[label], pic))
        self.data = contents

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # return self.data[i]
        return torch.LongTensor(self.data[i][0]), self.data[i][1], self.data[i][2]


class BertDataset(torch.utils.data.Dataset):

    def __init__(self, config, data_class):
        global path
        if data_class == 'train':
            path = config.train_path
        elif data_class == 'val':
            path = config.val_path
        elif data_class == 'test':
            path = config.test_path
        self.class_int_dict = {item: i for i, item in enumerate(config.class_list)}
        self.dataset = pd.read_csv(path, usecols=['path', 'cluster'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset.iloc[i]['path']
        label = self.class_int_dict[str(self.dataset.iloc[i]['cluster'])]
        return text, label


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
