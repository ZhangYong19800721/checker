# coding=utf-8

import random
import tools
import pyltp
import pickle

model_path = r"./ltp_data_v3.4.0/cws.model"
user_dict = r"./ltp_data_v3.4.0/userdict.txt"
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(model_path, user_dict)

corpus = []
file_name_template = r"./data/data{}.xlsx"
for n in range(1, 10+1):
    print(f"read data sheet {n}")
    file_name = file_name_template.format(n)
    data_part = tools.readRawData(file_name)  # 获取数据
    data_part = [x for x in data_part if not tools.isContainAttachment(x['body'])]  # 去除包含附件的数据条目
    sample_id = 0
    for article in data_part:
        if sample_id % 1000 == 0:
            print(f"process {sample_id}th sample")
        sample_id += 1
        article['body'] = list(segmentor.segment(article['body']))  # 分词
        article['body'] = tools.segmentRefine_PuctuationMark(article['body'])  # 对标点符号进行更多的处理
        article['body'] = tools.segmentRefine_ChineseDigit(article['body'])  # 对中文数字进行更多的处理
        article['body'] = tools.segmentRefine_Digit(article['body'])  # 对数字进行更多的处理
        article['body'] = tools.segmentRefine_English(article['body'])  # 对英文进行更多的处理
        article['body'] = ['SOS'] + [x for x in article['body'] if x != ""] + ['EOS']
    corpus += data_part

# 只保留标签为pass或reject的样本
print(f"read {len(corpus)} samples totally. only pass and reject samples will be maintained.")
corpus = [x for x in corpus if x['label'] == 'pass' or x['label'] == 'reject']
print(f"read {len(corpus)} samples remained.")

# 去掉数据一致但标签不一致的样本
print("remove unconsistent samples ......")
corpus.sort(key=lambda x: len(x['body']))  # 按照文章长度从小到大排序
sample_num = len(corpus)
removeSampleList = []  # 存放需要被删除的样本下标
for i in range(sample_num):
    if i % 10000 == 0:
        print(f"check {i}th sample")
    current_body_len = len(corpus[i]['body'])
    for j in range(i + 1, sample_num):
        if len(corpus[j]['body']) > current_body_len:
            break
        if corpus[i]['label'] != corpus[j]['label'] and corpus[i]['body'] == corpus[j]['body']:
            removeSampleList.append(i)
            removeSampleList.append(j)

removeSampleList = list(set(removeSampleList))
removeSampleList.sort(reverse=True)
print(f"{len(removeSampleList)} incosistent samples will be removed.")

for id in removeSampleList:
    corpus.pop(id)
print(f"DONE! {len(corpus)} samples remained.")

# 去掉内容重复的样本
print("remove redundent samples ......")
corpus.sort(key=lambda x: len(x['body']))
sample_num = len(corpus)
removeSampleList = []  # 存放需要被删除的样本下标
for i in range(sample_num):
    if i % 1000 == 0:
        print(f"check {i}th sample")
    current_body_len = len(corpus[i]['body'])
    for j in range(i + 1, sample_num):
        if len(corpus[j]['body']) > current_body_len:
            break
        if corpus[i]['body'] == corpus[j]['body']:
            removeSampleList.append(j)

removeSampleList = list(set(removeSampleList))
removeSampleList.sort(reverse=True)
print(f"{len(removeSampleList)} redundency samples will be removed.")
for id in removeSampleList:
    corpus.pop(id)
print(f"DONE! {len(corpus)} samples remained.")

# 去掉文章长度小于10，大于1000的样本
print("remove samples shorter than 10 and longer than 1000")
corpus = [x for x in corpus if 10 <= len(x['body']) and len(x['body']) <= 1000]
print(f"DONE! {len(corpus)} samples remained.")

# 分训练集和测试集
random.shuffle(corpus)  # 打乱顺序
trainset = corpus[:int(0.8 * len(corpus))]  # 80%的数据作为训练集
testset = corpus[int(0.8 * len(corpus)):]  # 20%的数据作为测试集

# 建词汇表
print("establish vocabulary ... ")
voc = tools.establishVocabulary(trainset, "党员网语料库词汇表")  # 第1次建词汇表
trainset = tools.splitUncommonWord(trainset, voc, 5)  # 如果一个多字词的出现次数少于5次，将它拆分为单字词
voc = tools.establishVocabulary(trainset, "党员网语料库词汇表")  # 第2次建词汇表
voc.trim(5)  # 去掉出现次数少于5次的词
print("DONE!")

# 保存词汇表
print("save vocabulary ... ")
vocabulary_filename = open(r"./data/vocabulary.voc", "wb")
pickle.dump(voc, vocabulary_filename)
vocabulary_filename.close()

# 保存训练集
print("save trainset ... ")
trainset_filename = open(r"./data/trainset.cps", "wb")
pickle.dump(trainset, trainset_filename)

# 保存测试集
print("save testset ... ")
testset_filename = open(r"./data/testset.cps", "wb")
pickle.dump(testset, testset_filename)

# 将训练集转换为数字形式
print("transform trainset to digit... ")
for n in range(len(trainset)):
    if n % 1000 == 0:
        print(f"transform trainset sample {n} to digit form")
    article = trainset[n]['body']
    article_digit = []
    for word in article:
        if word in voc.word2index:
            article_digit.append(voc.word2index[word])
        else:
            article_digit.append(voc.word2index["UNKNOWN"])
    trainset[n]['body'] = article_digit
    trainset[n]['label'] = 1 if trainset[n]['label'] == 'pass' else 0

# 将测试集转换为数字形式
print("transform testset to digit... ")
testset = tools.splitUnknownWord(testset, voc)  # 如果一个多字词从未在词表中出现，先将它拆分为单字词
for n in range(len(testset)):
    if n % 1000 == 0:
        print(f"transform testset sample {n} to digit form")
    article = testset[n]['body']
    article_digit = []
    for word in article:
        if word in voc.word2index:
            article_digit.append(voc.word2index[word])
        else:
            article_digit.append(voc.word2index["UNKNOWN"])
    testset[n]['body'] = article_digit
    testset[n]['label'] = 1 if testset[n]['label'] == 'pass' else 0

# 保存数字训练集
print("save trainset digit... ")
trainset_digitfilename = open(r"./data/trainset_digit.cps", "wb")
pickle.dump(trainset, trainset_digitfilename)

# 保存数字测试集
print("save testset digit... ")
testset_digitfilename = open(r"./data/testset_digit.cps", "wb")
pickle.dump(testset, testset_digitfilename)
