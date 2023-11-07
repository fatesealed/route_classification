# coding: UTF-8
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
import pickle as pkl
from train_eval import train, init_network, test
from importlib import import_module
from utils import CustomDataset

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

    dataset = CustomDataset(config)
    vocab = pkl.load(open(config.vocab_path, 'rb'))  # 打开词表
    # 定义拆分比例
    train_size = int(0.9 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # 使用random_split函数拆分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # # 加载训练集 验证集 测试集
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # # 训练
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)

    # 初始化参数
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_loader, val_loader, test_loader, notes)
    t = test(config, model, test_loader)
    # 打开文件，以“a”模式（追加模式）写入文本
    with open('ship_data/res.txt', 'a') as file:
        file.write(str(config.embed) + '_' + notes)
        # 追加文本内容到文件末尾
        file.write(config.is_random + ' ' + config.model_name + '\n')
        file.write(t)
        file.write('\n')
