import torch
import pickle
import random
import itertools
import tools
import numpy as np


class GCDYW(object):
    """
        共产党员网，自动审核训练数据集
    """

    # 构造函数
    # 参数
    # filename: 数据集文件名
    # transform: 数据预处理
    def __init__(self, filename):
        dataset_file = open(filename, "rb")  # 打开数据集文件
        dataset = pickle.load(dataset_file)  # 使用pickle.load读入数据集
        self.pos_set = [x for x in dataset if x['label'] == 1]
        self.neg_set = [x for x in dataset if x['label'] == 0]
        dataset_file.close()  # 关闭文件

    # 返回数据集大小
    def getLen(self):
        return len(self.pos_set), len(self.neg_set)

    def getPosItem(self, idx):
        return self.pos_set[idx]

    def getNegItem(self, idx):
        return self.neg_set[idx]

    def trim(self, minlen=20, maxlen=1000):
        self.pos_set = [x for x in self.pos_set if minlen <= len(x['article']) and len(x['article']) <= maxlen]
        self.neg_set = [x for x in self.neg_set if minlen <= len(x['article']) and len(x['article']) <= maxlen]


class LOADER(object):
    """
    数据加载器
    """

    def __init__(self, dataset, minibatch_size=100):
        self.dataset = dataset  # 数据集
        self.minibatch_size = minibatch_size  # batch的大小
        pos_len, neg_len = self.dataset.getLen()
        self.minibatch_num =  max(pos_len // (self.minibatch_size//2), neg_len // (self.minibatch_size//2))  # batch的数量

    def __len__(self):  # 返回batch的数量
        return self.minibatch_num

    def __getitem__(self, idx):
        pos_len, neg_len = self.dataset.getLen()
        select_pos_id = random.choices(range(pos_len), k=self.minibatch_size // 2)
        select_neg_id = random.choices(range(neg_len), k=self.minibatch_size // 2)
        minibatch = []
        for i in select_pos_id:
            minibatch.append(self.dataset.getPosItem(i))
        for i in select_neg_id:
            minibatch.append(self.dataset.getNegItem(i))

        minibatch_article = [x['article'] for x in minibatch]
        minibatch_article = tools.zeroPadding(minibatch_article)  # 较短的序列补零，调整时间方向为列方向，每一列表示一个数据样本
        minibatch_label = [x['label'] for x in minibatch]
        minibatch_rowid = [x['row_id'] for x in minibatch]
        minibatch_filename = [x['filename'] for x in minibatch]
        minibatch_keywords = [x['keywords'] for x in minibatch]


        minibatch_article = torch.LongTensor(minibatch_article)  # 将列表转换为张量
        minibatch_label = torch.LongTensor(minibatch_label)
        return {'article': minibatch_article, 'label': minibatch_label, 'row_id': minibatch_rowid, 'filename': minibatch_filename, 'keywords': minibatch_keywords}
