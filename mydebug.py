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
    # if data['filename'] == './data/data7.xlsx' and data['row_id'] == 43520:
    print(data)
    print(''.join([voc.index2word[x] for x in data['body']]))
    print("\n\n")

for i in range(10):
    data = trainset.getNegItem(i)
    # if data['filename'] == './data/data7.xlsx' and data['row_id'] == 43520:
    print(data)
    print(''.join([voc.index2word[x] for x in data['body']]))
    print("\n\n")

for i in range(10):
    data = testset.getPosItem(i)
    # if data['filename'] == './data/data7.xlsx' and data['row_id'] == 43520:
    print(data)
    print(''.join([voc.index2word[x] for x in data['body']]))
    print("\n\n")

for i in range(10):
    data = testset.getNegItem(i)
    # if data['filename'] == './data/data7.xlsx' and data['row_id'] == 43520:
    print(data)
    print(''.join([voc.index2word[x] for x in data['body']]))
    print("\n\n")

pos_num, neg_num = testset_origin.getLen()
for n in range(pos_num):
    if testset_origin.getPosItem(n)['row_id']==23999:
        print(testset_origin.getPosItem(n))

for n in range(neg_num):
    if testset_origin.getNegItem(n)['row_id']==23999:
        print(testset_origin.getNegItem(n))