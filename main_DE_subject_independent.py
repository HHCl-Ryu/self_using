import os
import sys
import numpy as np
import torch
import torch.optim as optim
from scipy import io as scio
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import zscore
from model import DGCNN
from utils import eegDataset
import copy

os.environ['TORCH_HOME'] = './'  # 设置环境变量

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_DE_SEED(load_path):
    filePath = load_path 
    datasets = scio.loadmat(filePath)
    DE = datasets['DE']
    dataAll = np.transpose(DE, [1,0,2])
    labelAll = datasets['labelAll'].flatten()
    labelAll = labelAll + 1
    return dataAll, labelAll

def load_dataloader(data_train, data_test, label_train, label_test):
    batch_size = 128  # 增加批量大小
    train_iter = DataLoader(dataset=eegDataset(data_train, label_train),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)
    test_iter = DataLoader(dataset=eegDataset(data_test, label_test),
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=1)
    return train_iter, test_iter

def train(train_iter, test_iter, model, criterion, optimizer, num_epochs, sub_name, patience=5):
    print('began training on', device, '...')
    acc_test_best = 0.0
    n = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for ep in range(num_epochs):
        model.train()
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_iter:
            images = images.float().to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels.long())
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(ep + 1, batch_id, total_loss / batch_id, accuracy))
            batch_id += 1
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))

        acc_test = evaluate(test_iter, model)
        if acc_test > acc_test_best:
            acc_test_best = acc_test
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = ep + 1
            n = 0  # 重置耐心计数器
        else:
            n += 1
            if n >= patience:
                print(f'Early stopping triggered at epoch {ep + 1}')
                break
        print('>>> best test Accuracy: {}'.format(acc_test_best))

    model.load_state_dict(best_model_weights)  # 加载最佳模型权重
    print(f'Best model found at epoch {best_epoch}')
    return acc_test_best

def evaluate(test_iter, model):
    print('began test on', device, '...')
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_iter:
            images = images.float().to(device)
            labels = labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
    print('test Accuracy: {}'.format(correct / total))
    return correct / total

def main_LOCV():
    dir = 'F:/EEG/git/seed/DGCNN-main/DE/session_all_4/'  # 修改路径以适应新的数据集
    file_list = os.listdir(dir)
    sub_num = len(file_list)
    xdim = [128, 4, 5]  # 修改为4个通道
    k_adj = 40
    num_out = 64
    num_epochs = 20  # 减少 epoch 数
    acc_mean = 0
    acc_all = []
    for sub_i in range(sub_num):
        load_path = dir + file_list[sub_i]
        data_test, label_test = load_DE_SEED(load_path)
        data_test = zscore(data_test)
        model = DGCNN(xdim, k_adj, num_out).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0001)  # 使用 AdamW 优化器，增加学习率
        train_list = copy.deepcopy(file_list)
        train_list.remove(file_list[sub_i])
        train_num = len(train_list)
        data_train = []
        label_train = []
        for train_i in range(train_num):
            train_path = dir + train_list[train_i]
            data, label = load_DE_SEED(train_path)
            data = zscore(data)
            if train_i == 0:
                data_train = data
                label_train = label
            else:
                data_train = np.concatenate((data_train, data), axis=0)
                label_train = np.concatenate((label_train, label), axis=0)
        data_train = torch.tensor(data_train)
        label_train = torch.tensor(label_train)
        print(f'Training data size: {data_train.size()}, Training labels size: {label_train.size()}')
        print(f'Test data size: {data_test.shape}, Test labels size: {label_test.shape}')
        train_iter, test_iter = load_dataloader(data_train, data_test, label_train, label_test)
        acc_test_best = train(train_iter, test_iter, model, criterion, optimizer, num_epochs, file_list[sub_i])
        acc_mean = acc_mean + acc_test_best / sub_num
        acc_all.append(acc_test_best)
    print('save...')
    scio.savemat('./result/acc_all/acc_de_SEED_LOCV.mat', {'acc_all': acc_all, 'sub_list': np.array(file_list, dtype=np.object)})
    print('>>> LOSV test acc: ', acc_all)
    print('>>> LOSV test mean acc: ', acc_mean)
    print('>>> LOSV test std acc: ', np.std(np.array(acc_all)))

if __name__ == '__main__':
    sys.exit(main_LOCV())