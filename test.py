import pickle
import random

corpus_file = open(r"D:\FTPROOT\workspace3\data\corpus.cps", "rb")
corpus = pickle.load(corpus_file)
corpus_file.close()

catch = [x for x in corpus if "颦" in x['body']]
print(catch)