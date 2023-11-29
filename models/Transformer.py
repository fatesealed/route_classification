import torch.nn as nn


class ModelConfig(object):
    """配置参数"""

    def __init__(self, notes=''):
        self.model_name = 'Transformer'
        self.save_path = f'./result/{self.model_name}_{notes}.pth'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name

        self.dropout = 0.5  # 随机失活
        self.learning_rate = 5e-4  # 学习率
        # self.dim_model = 1
        # self.hidden = 1024
        # self.last_hidden = 512
        self.num_head = 10
        self.num_encoder = 6


'''Attention Is All You Need'''


class Model(nn.Module):
    def __init__(self, model_config, data_config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            data_config.embedding_pretrained,
            freeze=False) if data_config.embedding_pretrained is not None else nn.Embedding(data_config.n_vocab,
                                                                                            data_config.embed,
                                                                                            padding_idx=data_config.n_vocab - 1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=data_config.embed, nhead=model_config.num_head,
                                                        batch_first=True)
        self.encoders = nn.TransformerEncoder(self.encoder_layer, num_layers=model_config.num_encoder)
        self.fc = nn.Linear(data_config.embed, data_config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = self.encoders(out)
        # out = out.reshape(out.size(0), -1)
        out = out[:, 0, :]
        out = self.fc(out)
        return out
