import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from DATASET_MNIST import *


class RNN_MODEL(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(RNN_MODEL, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = 0 if n_layers == 1 else dropout  # 当只有1层的时候不做dropout，当层数大于1层时使用输入的dropout参数

        # 初始化一个门单元GRU，输入维度input_size等于hidden_size，因为输入的是一个词向量，它的特征维度等于
        # hidden_size，n_layers指定了层数，bidirectional=True指定了采用双向的GRU，
        self.gru = nn.GRU(input_size, hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)

        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)  # gru的输出作为fc1的输入
        self.fc2 = nn.Linear(hidden_size, 10)  # fc1的输出作为fc2的输入

    def forward(self, input_seq, hidden=None):
        outputs_gru, hidden_gru = self.gru(input_seq, hidden)
        x = outputs_gru[-1, :, :self.hidden_size]  # 正向GRU最后的输出
        y = outputs_gru[+0, :, self.hidden_size:]  # 反向GRU最后的输出
        outputs_gru = torch.cat((x, y), 1)  # 将正向GRU末端的输出和反向GRU末端的输出拼接起来

        # 将GRU的输出送入一个全连接的Softmax判决层
        outputs = F.relu(self.fc1(outputs_gru))
        outputs = F.softmax(self.fc2(outputs), dim=1)
        return outputs


if __name__ == '__main__':
    start_time = time.time()

    # 如果GPU可用就使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rnn_model = RNN_MODEL(28, 500)  # 初始化一个RNN_MODEL的实例

    # 加载MNIST训练数据
    trainset = TRAINSET("./data/mnist.mat")
    trainset_loader = DATASET_LOADER(trainset, minibatch_size=100)

    # 目标函数CrossEntropy
    criterion = nn.CrossEntropyLoss()

    # 准备最优化算法
    optimizer = optim.SGD(rnn_model.parameters(), lr=0.001, momentum=0.9)

    rnn_model.to(device)
    aveloss = []
    for epoch in range(30):  # 对全部的训练数据进行n次遍历
        for minibatch_id in range(len(trainset_loader)):
            minibatch = trainset_loader[minibatch_id]
            images = minibatch["image"].to(device)
            labels = minibatch["label"].to(device)
            optimizer.zero_grad()
            output_data = rnn_model(images)
            loss = criterion(output_data, labels)
            aveloss.append(loss.item())
            while len(aveloss) > len(trainset_loader):
                aveloss.pop(0)
            ave_loss = sum(aveloss) / len(aveloss)
            print("epoch:%5d, batch_id:%5d, aveloss:%10.8f" % (epoch, minibatch_id, ave_loss))
            loss.backward()
            optimizer.step()  # Does the update

    end_time = time.time()
    print(f'train_time = {end_time - start_time}s')

    # 加载MNIST测试数据
    testset = TESTSET("./data/mnist.mat")
    testset_loader = DATASET_LOADER(testset, minibatch_size=100)

    error_count = 0
    with torch.no_grad():
        for minibatch_id in range(len(testset_loader)):
            minibatch = testset_loader[minibatch_id]
            images = minibatch["image"].to(device)
            lables = minibatch["label"]
            predict = rnn_model(images).to('cpu').numpy()
            predict = np.argmax(predict, axis=1)
            error_count += np.sum((predict != lables) + 0.0)

    error_rate = error_count / len(testset)
    print(f"error_rate = {error_rate}")
