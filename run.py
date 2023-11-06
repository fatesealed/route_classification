# coding: UTF-8
import torch
import numpy as np
# from train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# 选择模型
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# 选择是否使用预训练的词向量
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# 采用字模式还是词模式
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')

parser.add_argument('--notes', default='', type=str, help='note for this')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'ship_data'  # 数据集
    notes = args.notes

    embedding = 'embedding.npz'
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    if model_name == 'FastText':
        # 如果是fasttext 嵌入采用随机的方式
        from utils_fasttext import build_dataset, build_iterator
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    torch.backends.cudnn.benchmark = False

    print("加载数据")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    # 加载训练集 验证集 测试集
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    print("加载完毕")

    # # 训练
    # config.n_vocab = len(vocab)
    # model = x.Model(config).to(config.device)
    # if model_name != 'Transformer':
    #     # 初始化参数
    #     init_network(model)
    # print(model.parameters)
    # train(config, model, train_iter, dev_iter, test_iter, notes)
