# coding: UTF-8
import torch.nn as nn


class Config(object):
    """配置参数"""

    def __init__(self, notes=''):
        self.model_name = 'TextRNN'
        self.save_path = f'./result/{self.model_name}_{notes}.ckpt'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 10000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 100  # epoch数
        self.batch_size = 512  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率
        self.hidden_size = 256  # lstm隐藏层
        self.num_layers = 2  # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, model_config, data_config):
        super(Model, self).__init__()
        if data_config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(data_config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(data_config.n_vocab, data_config.embed, padding_idx=data_config.n_vocab - 1)
        self.lstm = nn.LSTM(data_config.embed, model_config.hidden_size, model_config.num_layers,
                            bidirectional=True, batch_first=True, dropout=model_config.dropout)
        self.fc = nn.Linear(model_config.hidden_size * 2, data_config.num_classes)

    def forward(self, x):
        out = self.embedding(x)  # [batch_size, seq_len, embeding]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

