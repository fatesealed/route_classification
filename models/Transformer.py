import math

import torch
import torch.nn as nn
from torch import Tensor


class ModelConfig(object):
    """配置参数"""

    def __init__(self, freeze, notes=''):
        self.freeze = freeze
        self.model_name = 'Transformer'
        self.save_path = f'./result/{self.model_name}_{notes}.pth'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name

        self.dropout = 0.5  # 随机失活
        # self.dim_model = 1
        # self.hidden = 1024
        # self.last_hidden = 512
        self.num_heads = 1
        self.num_encoder = 6


'''Attention Is All You Need'''


class Model(nn.Module):
    def __init__(self, model_config, data_config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            data_config.embedding_pretrained,
            freeze=model_config.freeze) if data_config.embedding_pretrained is not None else nn.Embedding(
            data_config.n_vocab,
            data_config.embed,
            padding_idx=data_config.n_vocab - 1)
        # self.pos_encoder = PositionalEncoding(data_config.embed)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=data_config.embed, nhead=model_config.num_heads,
                                                        batch_first=True)
        self.encoders = nn.TransformerEncoder(self.encoder_layer, num_layers=model_config.num_encoder)
        self.fc = nn.Linear(data_config.embed, data_config.num_classes)
        self.d_model = data_config.embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_model)
        # out = self.pos_encoder(out[:, :, :self.value])
        out = self.encoders(out)
        # out = out.reshape(out.size(0), -1)
        # 提取注意力权重矩阵
        # attention_weights = []
        # for layer in self.encoders.layers:
        #     attention_weights.append(layer.self_attn.attn.data.cpu().numpy())

        out = out[:, 0, :]
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
