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

voc_file = open(r"./data/vocabulary.voc", "rb")
voc = pickle.load(voc_file)
voc_file.close()

trainset = DATASET.GCDYW(r"./data/corpus_trainset_part1_digit.cps")  # 加载训练数据
dataloader = DATASET.LOADER(trainset, minibatch_size=20)  # 数据加载器，设定minibatch的大小

embedding_dim = 128  # 词向量的维度
hidden_size = 256
num_layers = 2
word_embedding = nn.Embedding(voc.num_words, embedding_dim)  # 初始化词向量
model = MODEL.ArticleReviewer(embedding_dim, hidden_size, word_embedding, num_layers)

criterion = nn.CrossEntropyLoss()  # 目标函数CrossEntropy
optimizer = optim.Adam(model.parameters())  # 准备最优化算法SGD

start_time = time.time()
model.to(device)  # 将模型推入GPU

lossList = []
exponentiaAveLoss = 1

minibatch_num = len(dataloader)
for epoch in range(20):
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
        exponentiaAveLoss = 99 / 100 * exponentiaAveLoss + 1 / 100 * loss
        aveLoss = sum(lossList) / len(lossList)  # 计算平均损失函数
        print("epoch:%5d, minibatch_id:%5d/%5d, expAveLoss:%10.8f, aveloss:%10.8f" % (epoch, minibatch_id, minibatch_num, exponentiaAveLoss, aveLoss))
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

end_time = time.time()
print(f'train_time = {(end_time - start_time) / 60} min')

model_file = open(r"./model/model.pkl", "wb")
pickle.dump(model, model_file)
model_file.close()
