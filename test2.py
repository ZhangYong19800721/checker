# coding=utf-8

import random
import tools
import pyltp

model_path = r'F:\workspace3\ltp_data_v3.4.0\ltp_data_v3.4.0\cws.model'
user_dict = r'F:\workspace3\ltp_data_v3.4.0\ltp_data_v3.4.0\userdict.txt'
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(model_path, user_dict)

data_file_name1 = r'F:\workspace3\data\data1-part1.xlsx'  # 路径字符串前加r，阻止字符串转义
data1 = tools.readRawData(data_file_name1)  # 获取数据

data1 = [x for x in data1 if not tools.isContainAttachment(x['body'])]
# data1_group1 = [x for x in data1 if x['label'] == 'pass']
# data1_group2 = [x for x in data1 if x['label'] == 'reject']

for n in data1:
    n['body'] = list(segmentor.segment(n['body']))
    n['body'] = tools.segmentRefine(n['body'])

voc = tools.Vocabulary("党员网")
for n in data1:
    voc.addSentence(n['body'])

L = [(v, k) for (k, v) in voc.word2count.items()]
L.sort(reverse=True)

output_file = open("output.txt", "w")
for v in L:
    output_file.write(v[1] + " " + str(v[0]) + "\n")
