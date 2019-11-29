import random
import tools
import pyltp
import pickle

# load the vocabulary
voc_file = open(r"./data/vocabulary.voc", "rb")
voc = pickle.load(voc_file)
voc_file.close()

origin_testset_file = open(r"./data/testset.cps", "rb")
testset = pickle.load(origin_testset_file)
origin_testset_file.close()

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

# 保存数字测试集
print("save testset digit... ")
testset_digitfilename = open(r"./data/testset_digit.cps", "wb")
pickle.dump(testset, testset_digitfilename)