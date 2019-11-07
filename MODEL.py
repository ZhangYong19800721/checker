# 自动审核模型

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArticleReviewer(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, num_layers=1, dropout=0):
        super(ArticleReviewer, self).__init__()  # 先调用基类的构造函数
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.dropout = dropout

        # 初始化一个门单元GRU，输入维度input_size等于hidden_size，因为输入的是一个词向量，它的特征维度等于
        # hidden_size，n_layers指定了层数，bidirectional=True指定了采用双向的GRU，
        # 当只有1层的时候不做dropout，当层数大于1层时使用输入的dropout参数
        self.gru = nn.GRU(input_size, hidden_size, self.num_layers, dropout=0 if self.num_layers==1 else self.dropout, bidirectional=True)

        self.dropout_fc1 = nn.Dropout(self.dropout)
        self.dropout_fc2 = nn.Dropout(self.dropout)

        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)  # gru的输出作为fc1的输入
        self.fc2 = nn.Linear(hidden_size, 2)  #

    def forward(self, article, article_len, hidden=None):
        # 先将词索引转换为词向量
        embedded = self.embedding(article)
        # 将数据送入GRU，并从末端获取输出
        packed = nn.utils.rnn.pack_padded_sequence(embedded, article_len, enforce_sorted=False)
        outputs_gru, hidden_gru = self.gru(packed, hidden)
        outputs_gru, _ = nn.utils.rnn.pad_packed_sequence(outputs_gru)

        x = outputs_gru[-1, :, :self.hidden_size]  # 正向GRU最后的输出
        y = outputs_gru[+0, :, self.hidden_size:]  # 反向GRU最后的输出
        outputs_gru = torch.cat((x, y), 1)  # 将正向GRU末端的输出和反向GRU末端的输出拼接起来

        # 将GRU的输出送入一个全连接的Softmax判决层
        outputs = torch.sigmoid(self.fc1(self.dropout_fc1(outputs_gru)))
        outputs = self.fc2(self.dropout_fc2(outputs))
        return outputs
