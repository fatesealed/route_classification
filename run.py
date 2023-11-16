# coding: UTF-8
import argparse
import pickle as pkl
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_eval import train, init_network
from utils import CustomDataset


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='船舶路径分类')
    parser.add_argument('--model', type=str, required=True,
                        help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--notes', default='', type=str, help='note for this')
    args = parser.parse_args()

    # 随机种子设置
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = 'ship_data'
    embedding = 'embedding.npz' if args.embedding == 'pre_trained' else 'random'
    notes = args.notes
    model_name = args.model

    # 动态导入模型配置和类
    model_module = import_module(f'models.{model_name}')
    config = model_module.Config(dataset, embedding)

    # 创建自定义数据集
    print('start read data...')
    train_dataset = CustomDataset(config, data_class='train')
    val_dataset = CustomDataset(config, data_class='val')
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    print('read data done...')

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 训练
    config.n_vocab = len(vocab)
    model = model_module.Model(config).to(config.device)

    # 初始化模型参数
    if model_name != 'Transformer':
        init_network(model)

    train(config, model, train_loader, val_loader, notes)


if __name__ == '__main__':
    main()
