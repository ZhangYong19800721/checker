# coding=utf-8

import pickle
import random
import tools

# 读入训练集
corpus_trainset_digit_file = open(r"D:\FTPROOT\workspace3\data\corpus_trainset_digit.cps", "rb")
corpus_trainset_digit = pickle.load(corpus_trainset_digit_file)
corpus_trainset_digit_file.close()


