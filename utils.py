# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
import time
from datetime import timedelta
import pandas as pd
from tqdm import tqdm

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(data_dir, tokenizer, max_size, min_freq):
    df = pd.read_csv(data_dir, usecols=['path'])
    vocab_dic = {}
    for index, row in tqdm(df.iterrows()):
        path_value = row['path']
        path_value = path_value.strip()
        for word in tokenizer(path_value):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                 :max_size]  # 排序 去除低频词
    word_to_id = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}  # 转换为从0开始
    word_to_id.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})  # 添加UNK和PAD
    return word_to_id


def build_dataset(config):
    tokenizer = lambda x: x.split('|')  # word-level
    vocab = pkl.load(open(config.vocab_path, 'rb'))  # 打开词表
    print(f"词表大小: {len(vocab)}")
    class_int_dict = {item:i + 1 for i, item in enumerate(config.class_list)}

    def load_dataset(data_dir, pad_size):
        df = pd.read_csv(data_dir, usecols=['path', 'cluster'])  # 读取csv
        contents = []
        for index, row in tqdm(df.iterrows()):
            content, label = row['path'], row['cluster']
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size: # 统一长度
                if seq_len < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            words_line = []
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, class_int_dict[label], seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    dataset = load_dataset(config.data_path, config.pad_size)
    return vocab, dataset


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
