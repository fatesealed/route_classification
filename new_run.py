# coding: UTF-8
import argparse
import pickle as pkl
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import models.ShipRNN as ShipRNN
from models.AlexNet import AlexNet
from models.LeNet import LeNet
from models.MyResNet import MyResNet
from models.ResNet18 import ResNet
from models.VGG import VGG, VGG16
from train_eval import train, init_network, test
from utils import CustomDataset, DataConfig


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='biSAMNet船舶路径分类')
    parser.add_argument('--dim', type=int, required=True, help='维度')
    parser.add_argument('--class_type', default='cluster', type=str, help='cluster or length_range')
    parser.add_argument('--comment', default=False, help='false or true')

    args = parser.parse_args()

    # 随机种子设置
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    embedding = 'word2vec'
    dim = args.dim
    model_name = 'ShipRNN'
    class_type = args.class_type
    notes = f'{dim}_{embedding}_{args.comment}'

    model_config = ShipRNN.ModelConfig(notes=notes)
    data_config = DataConfig(embedding, dim, class_type)
    print(model_name, data_config.dim, embedding)

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

    # model = ShipRNN.Model(model_config, data_config).to(data_config.device)
    model = MyResNet(data_config.num_classes)

    # 初始化模型参数

    init_network(model)
    summary(model, input_size=(2, 1, 31, 28), dtypes=[torch.float])
    train(model_config, data_config, model, train_loader, val_loader, notes)
    # 将测试结果写入文件
    res, res1 = test(data_config, model, test_loader, model_path=model_config.save_path)

    # 获取当前时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M")

    with open(f'res_6/{model_config.model_name}_{notes}_{formatted_time}_{embedding}.txt', "w") as file:
        file.write(str(res))
        file.write(str(res1))


if __name__ == '__main__':
    main()
