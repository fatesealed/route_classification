# coding: UTF-8
import argparse
import pickle as pkl
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from train_eval import train, init_network, test
from utils import CustomDataset, DataConfig


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='船舶路径分类')
    parser.add_argument('--model', type=str, required=True,
                        help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer, '
                             'BERT')
    parser.add_argument('--embedding', default='word2vec', type=str, help='random or word2vec or fasttext')
    parser.add_argument('--notes', default='', type=str, help='note for this')
    args = parser.parse_args()

    # 随机种子设置
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    embedding = 'random' if args.embedding == 'random' else args.embedding
    notes = args.notes
    model_name = args.model

    # 动态导入模型配置和类
    model_module = import_module(f'models.{model_name}')
    model_config = model_module.ModelConfig(notes)
    data_config = DataConfig(embedding)
    print(data_config.dim, embedding)

    # 创建自定义数据集
    print('start read data...')
    train_dataset = CustomDataset(data_config, data_class='train')
    val_dataset = CustomDataset(data_config, data_class='val')
    test_dataset = CustomDataset(data_config, data_class='test')
    vocab = pkl.load(open(data_config.vocab_path, 'rb'))
    data_config.n_vocab = len(vocab)
    print('read data done...')

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=data_config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=data_config.batch_size)

    # 训练

    model = model_module.Model(model_config, data_config).to(data_config.device)

    # 初始化模型参数
    if model_name != 'Transformer':
        init_network(model)
    summary(model, input_size=(2, 30), dtypes=[torch.long])
    train(model_config, data_config, model, train_loader, val_loader, notes)
    # 将测试结果写入文件
    res = test(data_config, model, test_loader, model_path=model_config.save_path)

    # 获取当前时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M")

    with open(f'res/{model_config.model_name}_{notes}_{formatted_time}_{embedding}.txt', "w") as file:
        file.write(str(res))


if __name__ == '__main__':
    main()
