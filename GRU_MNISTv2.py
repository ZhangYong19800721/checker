import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from DATASET_MNIST import *


class GRU_MODEL(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0):
        super(GRU_MODEL, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = 0 if num_layers == 1 else dropout  # 当只有1层的时候不做dropout，当层数大于1层时使用输入的dropout参数

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=self.dropout, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)  # 全连接层

    def forward(self, input_seq, hidden=None):
        outputs_gru, hidden_gru = self.gru(input_seq, hidden)
        x = outputs_gru[-1, :, :self.hidden_dim]  # 正向GRU最后的输出
        y = outputs_gru[+0, :, self.hidden_dim:]  # 反向GRU最后的输出
        outputs_gru = torch.cat((x, y), 1)  # 将正向GRU末端的输出和反向GRU末端的输出拼接起来
        # 将GRU的输出送入一个全连接的Softmax判决层
        outputs = F.relu(self.fc1(outputs_gru))
        outputs = self.fc2(outputs)
        return outputs


if __name__ == '__main__':
    start_time = time.time()

    # 如果GPU可用就使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_dim = 28
    hidden_dim = 512
    minibatch_size = 100
    gru_model = GRU_MODEL(input_dim, hidden_dim, num_layers=2, dropout=0.1)  # 初始化一个GRU_MODEL的实例

    # 加载MNIST训练数据
    trainset = TRAINSET("./data/mnist.mat")
    trainset_loader = DATASET_LOADER(trainset, minibatch_size=minibatch_size)

    # 目标函数CrossEntropy
    criterion = nn.CrossEntropyLoss()

    # 准备最优化算法
    optimizer = optim.Adam(gru_model.parameters())

    gru_model.to(device)
    aveloss = []

    epoch_num = 20
    minibatch_num = len(trainset_loader)
    for epoch in range(epoch_num):  # 对全部的训练数据进行n次遍历
        for minibatch_id in range(minibatch_num):
            minibatch = trainset_loader[minibatch_id]
            images = minibatch["image"].to(device)
            labels = minibatch["label"].to(device)
            optimizer.zero_grad()
            output_data = gru_model(images)
            loss = criterion(output_data, labels)
            aveloss.append(loss.item())
            while len(aveloss) > len(trainset_loader):
                aveloss.pop(0)
            ave_loss = sum(aveloss) / len(aveloss)
            print("epoch:%5d, minibatch:%5d/%d, loss:%10.8f, aveloss:%10.8f" % (epoch, minibatch_id, minibatch_num, loss.item(), ave_loss))
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(gru_model.parameters(), 100)
            optimizer.step()  # Does the update

    end_time = time.time()
    print(f'train_time = {end_time - start_time}s')

    # 加载MNIST测试数据
    testset = TESTSET("./data/mnist.mat")
    testset_loader = DATASET_LOADER(testset, minibatch_size=100)

    gru_model.eval()

    error_count = 0
    with torch.no_grad():
        for minibatch_id in range(len(testset_loader)):
            minibatch = testset_loader[minibatch_id]
            images = minibatch["image"].to(device)
            lables = minibatch["label"].numpy()
            predict = gru_model(images).to('cpu').numpy()
            predict = np.argmax(predict, axis=1)
            error_count += np.sum((predict != lables) + 0.0)

    error_rate = error_count / len(testset)
    print(f"error_rate = {error_rate}")
