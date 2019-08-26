import torch
import pickle
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
        self.dataset = pickle.load(dataset_file)  # 使用pickle.load读入数据集
        dataset_file.close()  # 关闭文件

    # 返回数据集大小
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class LOADER(object):
    """
    数据加载器
    """

    def __init__(self, dataset, batch_size=100):
        self.dataset = dataset  # 数据集
        self.batch_size = batch_size  # batch的大小
        self.batch_numb = len(self.dataset) // self.batch_size  # batch的数量

    def __len__(self):  # 返回batch的数量
        return self.batch_numb

    def __getitem__(self, idx):
        start, finish = self.batch_size * idx, self.batch_size * (idx + 1)
        minibatch = []
        for i in range(start, finish):
            minibatch.append(self.dataset[i])

        minibatch_article = [x['article'] for x in minibatch]
        minibatch_article = tools.zeroPadding(minibatch_article)  # 较短的序列补零，调整时间方向为列方向，每一列表示一个数据样本
        minibatch_label = [x['label'] for x in minibatch]

        minibatch_article = torch.LongTensor(minibatch_article)  # 将列表转换为张量
        # minibatch_articlelen = torch.LongTensor(minibatch_articlelen)
        minibatch_label = torch.LongTensor(minibatch_label)
        return {'article': minibatch_article, 'label': minibatch_label}
