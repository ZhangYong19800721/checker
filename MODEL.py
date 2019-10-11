# 自动审核模型

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArticleReviewer(nn.Module):
    def __init__(self, input_size, hidden_size, bottleneck_size, embedding, num_layers=1, dropout=0):
        super(ArticleReviewer, self).__init__()  # 先调用基类的构造函数
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.embedding = embedding
        self.dropout = 0 if num_layers == 1 else dropout  # 当只有1层的时候不做dropout，当层数大于1层时使用输入的dropout参数

        # 初始化一个门单元GRU，输入维度input_size等于hidden_size，因为输入的是一个词向量，它的特征维度等于
        # hidden_size，n_layers指定了层数，bidirectional=True指定了采用双向的GRU，
        self.gru1 = nn.GRU(input_size, hidden_size, self.num_layers, dropout=self.dropout, bidirectional=True)
        self.gru2 = nn.GRU(bottleneck_size, hidden_size, self.num_layers, dropout=self.dropout, bidirectional=True)

        self.dropout_fc0 = nn.Dropout(self.dropout)
        self.dropout_fc1 = nn.Dropout(self.dropout)

        self.fc0 = nn.Linear(2 * hidden_size, bottleneck_size)  # output of gru1 as input of fc0
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)  # output of gru2 as input of fc1
        self.fc2 = nn.Linear(hidden_size, 2)  #

    def forward(self, input_seq, sentence_len, article_len, hidden=None):
        # 先将词索引转换为词向量
        embedded = self.embedding(input_seq)
        # 将数据送入GRU，并从末端获取输出
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, sentence_len, enforce_sorted=False)
        outputs_gru1, hidden_gru1 = self.gru1(packed, hidden)
        outputs_gru1, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs_gru1)
		
        x1 = outputs_gru1[-1, :, :self.hidden_size]  # 正向GRU最后的输出
        y1 = outputs_gru1[+0, :, self.hidden_size:]  # 反向GRU最后的输出
        outputs_gru1 = torch.cat((x1, y1), 1)  # 将正向GRU末端的输出和反向GRU末端的输出拼接起来

        outputs_fc0 = torch.sigmoid(self.fc0(self.dropout_fc0(outputs_gru1)))

        # divide the outputs_gru1 into sub sequences.
        outputs_fc0 = list(outputs_fc0.split(article_len, dim=0))
        max_len = max(article_len)
        for i in range(len(outputs_fc0)):
            zeropadding = nn.ZeroPad2d((0, 0, 0, max_len - len(outputs_fc0[i])))
            outputs_fc0[i] = zeropadding(outputs_fc0[i])

        outputs_fc0 = torch.stack(outputs_fc0, dim=1)

        # print(article_len)
        packed = torch.nn.utils.rnn.pack_padded_sequence(outputs_fc0, article_len, enforce_sorted=False)
        outputs_gru2, hidden_gru2 = self.gru2(packed, hidden)
        outputs_gru2, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs_gru2)

        x2 = outputs_gru2[-1, :, :self.hidden_size]  # 正向GRU最后的输出
        y2 = outputs_gru2[+0, :, self.hidden_size:]  # 反向GRU最后的输出
        outputs_gru2 = torch.cat((x2, y2), 1)  # 将正向GRU末端的输出和反向GRU末端的输出拼接起来

        # 将GRU的输出送入一个全连接的Softmax判决层
        outputs = torch.sigmoid(self.fc1(self.dropout_fc1(outputs_gru2)))
        outputs = self.fc2(outputs)
        return outputs
