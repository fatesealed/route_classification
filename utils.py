# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
import time
from datetime import timedelta
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


# 自定义数据集类，需要实现__len__和__getitem__方法
class CustomDataset(Dataset):
    def __init__(self, config):
        tokenizer = lambda x: x.split('|')  # word-level
        vocab = pkl.load(open(config.vocab_path, 'rb'))  # 打开词表
        print(f"词表大小: {len(vocab)}")
        class_int_dict = {item: i for i, item in enumerate(config.class_list)}
        df = pd.read_csv(config.data_path, usecols=['path', 'cluster'])  # 读取csv
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
