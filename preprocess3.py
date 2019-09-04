import pickle
import random
import tools

# 读入训练集
corpus_trainset_file = open(r"./data/corpus_trainset.cps", "rb")
corpus_trainset = pickle.load(corpus_trainset_file)
corpus_trainset_file.close()

# 读入测试集
corpus_testset_file = open(r"./data/corpus_testset.cps", "rb")
corpus_testset = pickle.load(corpus_testset_file)
corpus_testset_file.close()

# 读入词汇表
vocabulary_file = open(r"./data/vocabulary.voc", "rb")
voc = pickle.load(vocabulary_file)  # 载入词汇表
vocabulary_file.close()

# 裁剪词汇表
voc.trim(50)

# 将词表中没有的词替换为UNKNOWN
corpus_trainset = tools.replaceUnknownWord(corpus_trainset, voc)
corpus_testset = tools.replaceUnknownWord(corpus_testset, voc)

# 将数据集转换为数字形式
corpus_trainset = tools.transformCorpusToDigit(corpus_trainset, voc)
corpus_testset = tools.transformCorpusToDigit(corpus_testset, voc)

# 将数字表示的训练集和测试集写入pkl文件
corpus_trainset_digit_file = open(r"./data/corpus_trainset_digit.cps", "wb")
corpus_testset_digit_file = open(r"./data/corpus_testset_digit.cps", "wb")
pickle.dump(corpus_trainset, corpus_trainset_digit_file)
pickle.dump(corpus_testset, corpus_testset_digit_file)
corpus_trainset_digit_file.close()
corpus_testset_digit_file.close()
print("END")
