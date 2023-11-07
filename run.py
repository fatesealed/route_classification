# coding: UTF-8
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader

from train_eval import train, init_network
from importlib import import_module
from utils import build_dataset, build_iterator

import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# 选择模型
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# 选择是否使用预训练的词向量
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')

parser.add_argument('--notes', default='', type=str, help='note for this')
args = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    torch.backends.cudnn.benchmark = False
    dataset = 'ship_data'  # 数据集
    notes = args.notes
    embedding = 'embedding.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    x = import_module('models.' + model_name)  # 动态导入对应训练类
    config = x.Config(dataset, embedding)  # 创建对应类的配置文件

    print("加载数据")
    vocab, dataset = build_dataset(config)
    # 定义拆分比例
    train_size = int(0.9 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # 使用random_split函数拆分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # # 加载训练集 验证集 测试集
    # train_iter = build_iterator(train_data, config)
    # dev_iter = build_iterator(dev_data, config)
    # test_iter = build_iterator(test_data, config)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    print("加载完毕")

    # # 训练
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    # 初始化参数
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    # train(config, model, train_loader, val_loader, test_loader, notes)
    for i, (x, y, _) in enumerate(train_loader):
        print(x,y,_)
        break
