# coding: UTF-8
import pickle as pkl
import time
from datetime import timedelta

import pandas as pd
import torch
from torch.utils.data import Dataset

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


# 自定义数据集类，需要实现__len__和__getitem__方法
class CustomDataset(Dataset):
    def __init__(self, config, data_class):
        tokenizer = lambda x: x.split('|')  # word-level
        if data_class == 'train':
            path = config.train_path
        elif data_class == 'val':
            path = config.val_path
        else:
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
