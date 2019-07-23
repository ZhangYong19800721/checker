# coding=utf-8

import random
import tools
import pyltp

model_path = r'D:\FTPROOT\workspace3\ltp_data_v3.4.0\cws.model'
user_dict = r'D:\FTPROOT\workspace3\ltp_data_v3.4.0\userdict.txt'
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(model_path, user_dict)

corpus = []
file_name_template = r"D:\FTPROOT\workspace3\data\data{}.xlsx"
for n in range(1, 10 + 1):
    file_name = file_name_template.format(n)
    data_part = tools.readRawData(file_name)  # 获取数据
    data_part = [x for x in data_part if not tools.isContainAttachment(x['body'])]  # 去除包含附件的数据条目
    for article in data_part:
        article['body'] = list(segmentor.segment(article['body']))  # 分词
        article['body'] = tools.segmentRefine_PuctuationMark(article['body'])  # 对标点符号进行更多的处理
        article['body'] = tools.segmentRefine_ChineseDigit(article['body'])  # 对中文数字进行更多的处理
        article['body'] = tools.segmentRefine_Digit(article['body'])  # 对数字进行更多的处理
        article['body'] = tools.segmentRefine_English(article['body'])  # 对英文进行更多的处理
        article['body'] = [x for x in article['body'] if x != ""]
    corpus += data_part

voc = tools.Vocabulary("党员网")  # 第1次建词汇表
for article in corpus:
    voc.addSentence(article['body'])

corpus = tools.Trim(corpus, voc, 4)

voc = tools.Vocabulary("党员网")  # 重建词表
for article in corpus:
    voc.addSentence(article['body'])

L1 = [(v, k) for (k, v) in voc.word2count.items()]
L1.sort(reverse=True)

output_file1 = open("output1.txt", "w", encoding='utf-8')
for v in L1:
    output_file1.write("{} {}\n".format(v[1], v[0]))

L2 = [(k, v) for (k, v) in voc.word2count.items()]
L2.sort()
output_file2 = open("output2.txt", "w", encoding='utf-8')
for v in L2:
    output_file2.write("{} {}\n".format(v[0], v[1]))
