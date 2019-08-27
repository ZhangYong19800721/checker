import pickle
import random

corpus_file = open(r"./data/corpus.cps", "rb")
corpus = pickle.load(corpus_file)
corpus_file.close()

corpus_positive = [x for x in corpus if x['label'] == 'pass']  # 通过的数据集
corpus_negative = [x for x in corpus if x['label'] == 'reject']  # 拒绝的数据集


# 从正例和反例中分别抽取10%的数据作为测试集
random.shuffle(corpus_positive)  # 打乱数据顺序
corpus_positive_testset = corpus_positive[:int(0.1 * len(corpus_positive))]
corpus_positive_trainset = corpus_positive[int(0.1 * len(corpus_positive)):]
random.shuffle(corpus_negative)  # 打乱数据顺序
corpus_negative_testset = corpus_negative[:int(0.1 * len(corpus_negative))]
corpus_negative_trainset = corpus_negative[int(0.1 * len(corpus_negative)):]

# 得到测试集和训练集
corpus_testset = corpus_positive_testset + corpus_negative_testset
random.shuffle(corpus_testset)
corpus_trainset = corpus_positive_trainset + corpus_negative_trainset
random.shuffle(corpus_trainset)

corpus_trainset_file = open(r"./data/corpus_trainset.cps", "wb")
corpus_testset_file = open(r"./data/corpus_testset.cps", "wb")
pickle.dump(corpus_trainset, corpus_trainset_file)
pickle.dump(corpus_testset, corpus_testset_file)
corpus_trainset_file.close()
corpus_testset_file.close()
