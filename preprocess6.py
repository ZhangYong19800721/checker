# coding=utf-8

import pickle
import random

trainset_file = open(r"./data/corpus_trainset_digit.cps", "rb")  # 打开数据集文件
trainset = pickle.load(trainset_file)
trainset_file.close()  # 关闭文件

trainset.sort(key=lambda x: len(x['article']))
sample_num = len(trainset)

rmlist = []  # 存放需要被删除的样本下标
for i in range(sample_num):
    print(i)
    current_article_len = len(trainset[i]['article'])
    for j in range(i + 1, sample_num):
        if len(trainset[j]['article']) > current_article_len:
            break
        if trainset[i]['label'] != trainset[j]['label'] and trainset[i]['article'] == trainset[j]['article']:
            rmlist.append(i)
            rmlist.append(j)
            # print(trainset[i], "\n\n", trainset[j], "\n\n")
            # print("-"*300)

rmlist = list(set(rmlist))
rmlist.sort(reverse=True)
print(f"{len(rmlist)} incosistent samples will be removed.")

for id in rmlist:
    trainset.pop(id)

random.shuffle(trainset)
trainset_file_consistent = open(r"./data/corpus_trainset_consistent_digit.cps", "wb")  # 打开数据集文件
pickle.dump(trainset, trainset_file_consistent)
trainset_file_consistent.close()  # 关闭文件

print("END")
