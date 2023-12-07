# coding: UTF-8
import argparse
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import AutoTokenizer

from train_eval import bert_train, test
from utils import BertDataConfig, BertDataset


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='船舶路径分类')
    parser.add_argument('--model', type=str, required=True,
                        help='BERT')
    parser.add_argument('--notes', default='', type=str, help='note for this')
    args = parser.parse_args()

    # 随机种子设置
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    notes = args.notes
    model_name = args.model

    # 动态导入模型配置和类
    model_module = import_module(f'models.{model_name}')
    model_config = model_module.ModelConfig(notes)
    data_config = BertDataConfig()

    # 创建自定义数据集
    print('start read data...')
    train_dataset = BertDataset(data_config, data_class='train')
    val_dataset = BertDataset(data_config, data_class='val')
    test_dataset = BertDataset(data_config, data_class='test')
    print('read data done...')

    # 加载字典和分词工具
    local_model_path = "/wzs/code/hg_model/dataroot/models/bert-base-chinese"
    token = AutoTokenizer.from_pretrained(local_model_path)

    def collate_fn(data):
        sents = [i[0] for i in data]
        labels = [i[1] for i in data]

        # 编码
        data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=30,
                                       return_tensors='pt',
                                       return_length=True)

        # input_ids:编码之后的数字
        # attention_mask:是补零的位置是0,其他位置是1
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']
        labels = torch.LongTensor(labels)
        return input_ids, attention_mask, token_type_ids, labels

    # # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=data_config.batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=data_config.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=data_config.batch_size, collate_fn=collate_fn)

    # 训练

    model = model_module.Model(data_config).to(data_config.device)

    summary(model, [(1, 30), (1, 30), (1, 30)], dtypes=[torch.long, torch.long, torch.long])
    bert_train(model_config, data_config, model, train_loader, val_loader, notes)
    # 将测试结果写入文件
    res = test(data_config, model, test_loader, model_path=model_config.save_path)

    # 获取当前时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M")

    with open(f'res/{model_config.model_name}_{notes}_{formatted_time}.txt', "w") as file:
        file.write(str(res))


if __name__ == '__main__':
    main()
