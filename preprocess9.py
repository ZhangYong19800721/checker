# coding=utf-8

import pickle
import random

trainset_file = open(r"./data/corpus_testset_consistent_digit.cps", "rb")  # 打开数据集文件
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
        if trainset[i]['article'] == trainset[j]['article']:
            rmlist.append(j)

rmlist = list(set(rmlist))
rmlist.sort(reverse=True)
print(f"{len(rmlist)} redundency samples will be removed.")

for id in rmlist:
    trainset.pop(id)

random.shuffle(trainset)
trainset_file_consistent_rmrepeat = open(r"./data/corpus_testset_consistent_rmrepeat_digit.cps", "wb")  # 打开数据集文件
pickle.dump(trainset, trainset_file_consistent_rmrepeat)
trainset_file_consistent_rmrepeat.close()  # 关闭文件

print("END")
