from importlib import import_module

from torch.utils.data import DataLoader

from train_eval import test
from utils import CustomDataset


def main(model_name='TextCNN', model_path='./result/TextCNN_one_embedding_layer.ckpt', dataset='ship_data', embedding='embedding.npz'):
    model_module = import_module(f'models.{model_name}')
    config = model_module.Config(dataset, embedding)
    test_dataset = CustomDataset(config, data_class='test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = model_module.Model(config).to(config.device)

    test(config, model, test_loader, model_path)


if __name__ == '__main__':
    main()
