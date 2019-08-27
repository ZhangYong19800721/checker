# coding=utf-8

import pickle
import random
import tools

# 读入训练集
trainset_file = open(r"./data/corpus_trainset.cps", "rb")
trainset = pickle.load(trainset_file)
trainset_file.close()

trainset_1 = [x for x in trainset if x['label']=='pass']
trainset_0 = [x for x in trainset if x['label']=='reject']

x1 = int((1/3) * len(trainset_1))
x2 = int((2/3) * len(trainset_1))
y1 = int((1/3) * len(trainset_0))
y2 = int((2/3) * len(trainset_0))
trainset_1_part1 = trainset_1[:x1]
trainset_1_part2 = trainset_1[x1:x2]
trainset_1_part3 = trainset_1[x2:]
trainset_0_part1 = trainset_0[:y1]
trainset_0_part2 = trainset_0[y1:y2]
trainset_0_part3 = trainset_0[y2:]

trainset_part1 = trainset_1_part1 + trainset_0_part1
trainset_part2 = trainset_1_part2 + trainset_0_part2
trainset_part3 = trainset_1_part3 + trainset_0_part3
random.shuffle(trainset_part1)
random.shuffle(trainset_part2)
random.shuffle(trainset_part3)

trainset_part1_file = open(r"./data/corpus_trainset_part1.cps", "wb")
trainset_part2_file = open(r"./data/corpus_trainset_part2.cps", "wb")
trainset_part3_file = open(r"./data/corpus_trainset_part3.cps", "wb")
pickle.dump(trainset_part1, trainset_part1_file)
pickle.dump(trainset_part2, trainset_part2_file)
pickle.dump(trainset_part3, trainset_part3_file)
trainset_part1_file.close()
trainset_part2_file.close()
trainset_part3_file.close()




