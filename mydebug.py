import pickle
import DATASET
import torch
import numpy as np

vocabulary_file = open(r"./data/vocabulary.voc", "rb")
voc = pickle.load(vocabulary_file)  # load vocabulary
vocabulary_file.close()

trainset = DATASET.GCDYW(r"./data/trainset_digit.cps")  # 加载训练数据
testset = DATASET.GCDYW(r"./data/testset_digit.cps")  # 加载训练数据
trainset_origin = DATASET.GCDYW(r"./data/trainset.cps")  # 加载训练数据
testset_origin = DATASET.GCDYW(r"./data/testset.cps")  # 加载训练数据

pos_num, neg_num = trainset.getLen()
for i in range(10):
    data = trainset.getPosItem(i)
    print(data)
    print(''.join([voc.index2word[x] for x in data['body']]))
    print("\n\n")

for i in range(10):
    data = trainset.getNegItem(i)
    print(data)
    print(''.join([voc.index2word[x] for x in data['body']]))
    print("\n\n")

for i in range(10):
    data = testset.getPosItem(i)
    print(data)
    print(''.join([voc.index2word[x] for x in data['body']]))
    print("\n\n")

for i in range(10):
    data = testset.getNegItem(i)
    print(data)
    print(''.join([voc.index2word[x] for x in data['body']]))
    print("\n\n")