# coding=utf-8

import time
import math
import torch  # 导入torch模块
import torch.nn as nn
import torch.optim as optim
import DATASET
import MODEL
import pickle
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./runs/training")

# load the vocabulary
voc_file = open(r"./data/vocabulary.voc", "rb")
voc = pickle.load(voc_file)
voc_file.close()

# 设定计算设备，当有GPU的时候使用GPU，否则使用CPU
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

log_file = open("./log/train_log%s.txt"%time.time(), "w", encoding='utf-8')

trainset = DATASET.GCDYW(r"./data/trainset_digit.cps")  # 加载训练数据
trainset.trim(20, 1000)
minibatch_size = 60
dataloader = DATASET.LOADER(trainset, minibatch_size=minibatch_size)  # 数据加载器，设定minibatch的大小
# dataloader = DATASET.TEST_LOADER(trainset, minibatch_size=minibatch_size)  # 数据加载器，设定minibatch的大小

embedding_dim = 100
hidden_size = 1500
num_layers = 1
dropout = 0.1
update_period = 1

word_embedding = nn.Embedding(voc.num_words, embedding_dim)  # 初始化词向量
model = MODEL.ArticleReviewer(embedding_dim, hidden_size, word_embedding, num_layers=num_layers, dropout=dropout)

start_epoch_id = 0
end_epoch_id = 100
try:
    model_pre_file_name = r"./model/model%03d.pkl"%(start_epoch_id-1)
    model_pre_file = open(model_pre_file_name, "rb")
    model = pickle.load(model_pre_file)
    if start_epoch_id > 0:
        print("加载上一次的模型", model_pre_file_name)
except:
    if start_epoch_id > 0:
        print("无法加载上一次的模型，将重新开始训练！")

criterion = nn.CrossEntropyLoss()  # 目标函数CrossEntropy
optimizer = optim.Adam(model.parameters())  # 准备最优化算法Adam
# optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)  # 准备最优化算法SGD

model.to(device)  # 将模型推入GPU
model.train()  # set Dropout layer to train mode

lossList = []

minibatch_num = len(dataloader)
# minibatch_num = 300

for epoch in range(start_epoch_id, end_epoch_id):
    start_time = time.time()
    optimizer.zero_grad()  # 梯度置零
    for minibatch_id in range(1, minibatch_num+1):
        minibatch = dataloader[minibatch_id]
        article = minibatch['article'].to(device)  # 将数据推送到GPU
        article_len = minibatch['article_len']
        label = minibatch['label'].to(device)  # 将数据推送到GPU
        model_output, prob = model(article, article_len)
        loss = criterion(model_output, label)
        lossList.append(loss)
        while len(lossList) > minibatch_num:
            lossList.pop(0)
        shortAveLoss = sum(lossList[-100:]) / min(len(lossList), 100)
        aveLoss = sum(lossList) / len(lossList)  # 计算平均损失函数
        loss.backward()  # 反向传播

        if minibatch_id % update_period == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 50)  # 限制梯度范数，避免梯度爆炸
            log_message = "epoch:%5d/%d, minibatch_id:%5d/%d, loss:%10.8f, shortAveLoss:%10.8f, aveLoss:%10.8f, grad_norm:%10f, time:%10d" % (
                epoch, end_epoch_id, minibatch_id, minibatch_num, loss, shortAveLoss, aveLoss, grad_norm, int(time.time() - start_time))
            print(log_message)
            log_file.write(log_message + "\n")
            if math.isnan(grad_norm):
                pass
            else:
                optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 梯度置零
        else:
            log_message = "epoch:%5d/%d, minibatch_id:%5d/%d, loss:%10.8f" % (
                epoch, end_epoch_id, minibatch_id, minibatch_num, loss)
            print(log_message)

        writer.add_scalar("CurrentLoss", loss.item(), epoch * minibatch_num + minibatch_id)
        writer.add_scalar("ShortAveLoss", shortAveLoss.item(), epoch * minibatch_num + minibatch_id)
        writer.add_scalar("LongAveLoss", aveLoss.item(), epoch * minibatch_num + minibatch_id)

    log_file.flush()
    writer.flush()

    # save model every epoch
    model_file = open(r"./model/model%03d.pkl" % epoch, "wb")
    pickle.dump(model, model_file)
    model_file.close()
    end_time = time.time()
    print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')

log_file.close()
