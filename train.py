# coding=utf-8

import time
import math
import torch  # 导入torch模块
import torch.nn as nn
import torch.optim as optim
import DATASET
import MODEL
import pickle

# 设定计算设备，当有GPU的时候使用GPU，否则使用CPU
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

log_file = open("log.txt", "w", encoding='utf-8')

voc_file = open(r"./data/vocabulary.voc", "rb")
voc = pickle.load(voc_file)
voc_file.close()

trainset = DATASET.GCDYW(r"./data/trainset_digit.cps")  # 加载训练数据
trainset.trim(20, 400)
minibatch_size = 30
dataloader = DATASET.LOADER(trainset, minibatch_size=minibatch_size)  # 数据加载器，设定minibatch的大小

embedding_dim = 50
bottleneck_dim = 200
hidden_size = 600
num_layers = 1
dropout = 0
update_period = 1

word_embedding = nn.Embedding(voc.num_words, embedding_dim)  # 初始化词向量
model = MODEL.ArticleReviewer(embedding_dim, hidden_size, bottleneck_dim, word_embedding, num_layers=num_layers, dropout=dropout)

try:
    model_pre_file = open(r"./model/model_pre.pkl", "rb")
    model = pickle.load(model_pre_file)
except:
    pass

criterion = nn.CrossEntropyLoss()  # 目标函数CrossEntropy
optimizer = optim.Adam(model.parameters())  # 准备最优化算法Adam
# optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)  # 准备最优化算法SGD

model.to(device)  # 将模型推入GPU
model.train()  # set Dropout layer to train mode

lossList = []

minibatch_num = len(dataloader)
# minibatch_num = 300
epoch_num = 100
for epoch in range(epoch_num):
    start_time = time.time()
    for minibatch_id in range(minibatch_num):
        minibatch = dataloader[minibatch_id]
        datas = minibatch['article'].to(device)  # 将数据推送到GPU
        sentence_len = minibatch['sentence_len']
        article_len = minibatch['article_len']
        label = minibatch['label'].to(device)  # 将数据推送到GPU
        if minibatch_id % update_period == 0:
            optimizer.zero_grad()  # 梯度置零
        model_output = model(datas, sentence_len, article_len)
        loss = criterion(model_output, label)
        lossList.append(loss)
        while len(lossList) > minibatch_num:
            lossList.pop(0)
        shortAveLoss = sum(lossList[-100:]) / min(len(lossList), 100)
        aveLoss = sum(lossList) / len(lossList)  # 计算平均损失函数
        loss.backward()  # 反向传播
        if minibatch_id % update_period == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100)  # 限制梯度范数，避免梯度爆炸
            log_message = "epoch:%5d/%d, minibatch_id:%5d/%d, loss:%10.8f, shortAveLoss:%10.8f, aveLoss:%10.8f, grad_norm:%20.18f" % (
                epoch, epoch_num, minibatch_id, minibatch_num, loss, shortAveLoss, aveLoss, grad_norm)
            print(log_message)
            log_file.write(log_message + "\n")
            if math.isnan(grad_norm):
                pass
            else:
                optimizer.step()  # 更新参数

    # save model every epoch
    model_file = open(r"./model/model%03d.pkl" % epoch, "wb")
    pickle.dump(model, model_file)
    model_file.close()
    end_time = time.time()
    print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')

log_file.close()
