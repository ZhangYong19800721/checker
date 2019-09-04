# coding=utf-8

import pickle
import time
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

trainset = DATASET.GCDYW(r"./data/corpus_trainset_consistent_rmrepeat_digit.cps")  # 加载训练数据
trainset.trim(20, 500)
# trainset.balance()
minibatch_size = 50
dataloader = DATASET.LOADER(trainset, minibatch_size=minibatch_size)  # 数据加载器，设定minibatch的大小

embedding_dim = 100
hidden_size = 800
num_layers = 2
dropout = 0.1

word_embedding = nn.Embedding(voc.num_words, embedding_dim)  # 初始化词向量
model = MODEL.ArticleReviewer(embedding_dim, hidden_size, word_embedding, num_layers=num_layers, dropout=dropout)

try:
    model_pre_file = open(r"./model/model_pre.pkl", "rb")
    model = pickle.load(model_pre_file)
except:
    pass

criterion = nn.CrossEntropyLoss()  # 目标函数CrossEntropy
optimizer = optim.Adam(model.parameters())  # 准备最优化算法Adam
# optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)  # 准备最优化算法SGD

start_time = time.time()
model.to(device)  # 将模型推入GPU

lossList = []
expAveLoss = 0

minibatch_num = len(dataloader)
# minibatch_num = 300
epoch_num = 100
for epoch in range(epoch_num):
    for minibatch_id in range(minibatch_num):
        minibatch = dataloader[minibatch_id]
        datas = minibatch['article'].to(device)  # 将数据推送到GPU
        label = minibatch['label'].to(device)  # 将数据推送到GPU
        optimizer.zero_grad()  # 梯度置零
        model_output = model(datas)
        loss = criterion(model_output, label)
        lossList.append(loss)
        while len(lossList) > minibatch_num:
            lossList.pop(0)
        expAveLoss = 99 / 100 * expAveLoss + 1 / 100 * loss
        aveLoss = sum(lossList) / len(lossList)  # 计算平均损失函数
        log_message = "epoch:%5d/%d, minibatch_id:%5d/%d, loss:%10.8f, expAveLoss:%10.8f, aveloss:%10.8f" % (
        epoch, epoch_num, minibatch_id, minibatch_num, loss, expAveLoss, aveLoss)
        print(log_message)
        log_file.write(log_message + "\n")
        loss.backward()  # 反向传播
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 100)  # 限制梯度范数，避免梯度爆炸
        optimizer.step()  # 更新参数

    # save model every epoch
    model_file = open(r"./model/model%03d.pkl" % epoch, "wb")
    pickle.dump(model, model_file)
    model_file.close()

end_time = time.time()
log_file.close()
print(f'train_time = {(end_time - start_time) / 60} min')
