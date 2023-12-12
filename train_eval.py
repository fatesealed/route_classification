# coding: UTF-8
import time

import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

from utils import get_time_dif


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:  # 如果不是嵌入层
            if 'weight' in name:  # weight 三种初始化方式
                if method == 'xavier' and len(w.size()) >= 2:
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:  # bias 置0
                nn.init.constant_(w, 0)
            else:
                pass


def train(model_config, data_config, model, train_iter, dev_iter, notes):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=data_config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch下标
    flag = False  # 记录是否很久没有效果提升
    model.train()
    writer_time = time.strftime('%m-%d_%H.%M', time.localtime())
    writer = SummaryWriter(log_dir=f'{model_config.log_path}/{str(data_config.embed)}_{notes}_{writer_time}')
    for epoch in range(data_config.num_epochs):
        print(f'Epoch [{epoch + 1}/{data_config.num_epochs}]')
        for x, y, _ in train_iter:
            x = x.to(data_config.device)
            y = y.to(data_config.device)
            outputs = model(x)
            loss = f.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = y.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                # train_recall = metrics.recall_score(true, predic)
                # train_f1 = metrics.f1_score(true, predic)
                results = evaluate(data_config, model, dev_iter)
                # Access accuracy and loss from the results dictionary
                dev_acc = results['accuracy']
                dev_loss = results['loss']
                if dev_loss < dev_best_loss:
                    # 当验证集损失下降时
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), model_config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {' \
                      '4:>6.2%},  Time: {5} {6} '
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > data_config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        scheduler.step()  # 学习率衰减 一轮衰减0.1
        if flag:
            break
    writer.close()


def bert_train(model_config, data_config, model, train_iter, dev_iter, notes):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=data_config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch下标
    flag = False  # 记录是否很久没有效果提升
    model.train()
    writer_time = time.strftime('%m-%d_%H.%M', time.localtime())
    writer = SummaryWriter(log_dir=f'{model_config.log_path}/{notes}_{writer_time}')
    for epoch in range(data_config.num_epochs):
        print(f'Epoch [{epoch + 1}/{data_config.num_epochs}]')
        for input_ids, attention_mask, token_type_ids, y in train_iter:
            input_ids = input_ids.to(data_config.device)
            attention_mask = attention_mask.to(data_config.device)
            token_type_ids = token_type_ids.to(data_config.device)
            y = y.to(data_config.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = f.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = y.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                # train_recall = metrics.recall_score(true, predic)
                # train_f1 = metrics.f1_score(true, predic)
                results = bert_evaluate(data_config, model, dev_iter)
                # Access accuracy and loss from the results dictionary
                dev_acc = results['accuracy']
                dev_loss = results['loss']
                if dev_loss < dev_best_loss:
                    # 当验证集损失下降时
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), model_config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {' \
                      '4:>6.2%},  Time: {5} {6} '
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > data_config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        scheduler.step()  # 学习率衰减 一轮衰减0.1
        if flag:
            break
    writer.close()


def test(config, model, test_iter, model_path, is_bert=False):
    # 选用表现最好的那轮
    model.load_state_dict(torch.load(model_path))
    model.eval()
    result = bert_evaluate(config, model, test_iter, is_test=True) if is_bert else evaluate(config, model, test_iter,
                                                                                            is_test=True)
    test_acc = result['accuracy']
    test_loss = result['loss']
    test_report = result['report']
    test_confusion = result['confusion_matrix']
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    return test_report


def bert_evaluate(config, model, data_iter, is_test=False):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        predict_all = []
        labels_all = []
        for input_ids, attention_mask, token_type_ids, y in data_iter:
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            token_type_ids = token_type_ids.to(config.device)
            y = y.to(config.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = f.cross_entropy(outputs, y)
            loss_total += loss.item()  # 使用 item() 获取标量损失值
            predict = torch.max(outputs, 1)[1]
            labels_all.append(y)
            predict_all.append(predict)

        labels_all = torch.cat(labels_all).cpu().numpy()  # 合并标签
        predict_all = torch.cat(predict_all).cpu().numpy()  # 合并预测

        acc = metrics.accuracy_score(labels_all, predict_all)

        results = {
            'accuracy': acc,
            'loss': loss_total / len(data_iter),
        }

        if is_test:
            results['report'] = metrics.classification_report(labels_all, predict_all, target_names=config.class_list,
                                                              digits=4)
            results['confusion_matrix'] = metrics.confusion_matrix(labels_all, predict_all)

        return results


def evaluate(config, model, data_iter, is_test=False):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        predict_all = []
        labels_all = []
        for x, y, _ in data_iter:
            x = x.to(config.device)
            y = y.to(config.device)
            outputs = model(x)
            loss = f.cross_entropy(outputs, y)
            loss_total += loss.item()  # 使用 item() 获取标量损失值
            predict = torch.max(outputs, 1)[1]
            labels_all.append(y)
            predict_all.append(predict)

        labels_all = torch.cat(labels_all).cpu().numpy()  # 合并标签
        predict_all = torch.cat(predict_all).cpu().numpy()  # 合并预测

        acc = metrics.accuracy_score(labels_all, predict_all)

        results = {
            'accuracy': acc,
            'loss': loss_total / len(data_iter),
        }

        if is_test:
            results['report'] = metrics.classification_report(labels_all, predict_all, target_names=config.class_list,
                                                              digits=4)
            results['confusion_matrix'] = metrics.confusion_matrix(labels_all, predict_all)

        return results
