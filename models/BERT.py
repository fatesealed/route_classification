# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel


class ModelConfig(object):
    """配置参数"""

    def __init__(self, notes=''):
        self.model_name = 'BERT'
        self.save_path = f'./result/{self.model_name}_{notes}.pth'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name


class Model(nn.Module):
    def __init__(self, data_config):
        super(Model, self).__init__()
        # 加载预训练模型
        local_model_path = "/wzs/code/hg_model/dataroot/models/bert-base-chinese"
        self.pretrained = BertModel.from_pretrained(local_model_path)
        # 不训练,不需要计算梯度
        for param in self.pretrained.parameters():
            param.requires_grad_(False)
        self.fc = torch.nn.Linear(768, data_config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        return out
