# coding=utf-8

import pickle
import random
import tools

# 读入词汇表
voc_file = open(r"./data/vocabulary.voc", "rb")
voc = pickle.load(voc_file)
voc_file.close()

# 读入训练集-part1
trainset_file_part1 = open(r"./data/corpus_trainset_part1.cps", "rb")
trainset_part1 = pickle.load(trainset_file_part1)
trainset_file_part1.close()
trainset_part1 = tools.transformCorpusToDigit(trainset_part1,voc)
trainset_file_part1_digit_file = open(r"./data/corpus_trainset_part1_digit.cps", "wb")
pickle.dump(trainset_part1,trainset_file_part1_digit_file)
trainset_file_part1_digit_file.close()

# 读入训练集-part2
trainset_file_part2 = open(r"./data/corpus_trainset_part2.cps", "rb")
trainset_part2 = pickle.load(trainset_file_part2)
trainset_file_part2.close()
trainset_part2 = tools.transformCorpusToDigit(trainset_part2,voc)
trainset_file_part2_digit_file = open(r"./data/corpus_trainset_part2_digit.cps", "wb")
pickle.dump(trainset_part2,trainset_file_part2_digit_file)
trainset_file_part2_digit_file.close()

# 读入训练集-part3
trainset_file_part3 = open(r"./data/corpus_trainset_part3.cps", "rb")
trainset_part3 = pickle.load(trainset_file_part3)
trainset_file_part3.close()
trainset_part3 = tools.transformCorpusToDigit(trainset_part3,voc)
trainset_file_part3_digit_file = open(r"./data/corpus_trainset_part3_digit.cps", "wb")
pickle.dump(trainset_part3,trainset_file_part3_digit_file)
trainset_file_part3_digit_file.close()

print("END")