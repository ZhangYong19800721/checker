# coding=utf-8

import pickle
import DATASET
import torch
import numpy as np

# 读入词汇表
vocabulary_file = open(r"./data/vocabulary.voc", "rb")
voc = pickle.load(vocabulary_file)  # 载入词汇表
vocabulary_file.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_file = open(r"./model/model_final.pkl", "rb")
model = pickle.load(model_file)
model.to(device)
model.eval()
model_file.close()

testset = DATASET.GCDYW(r"./data/corpus_trainset_consistent_rmrepeat_digit.cps")  # load the test data
testset.trim(20, 300)
minibatch_size = 100
dataloader = DATASET.LOADER(testset, minibatch_size=minibatch_size)  # set the minibatch size
minibatch_num = len(dataloader)

error_count, error_pos_count, error_neg_count, pos_num, neg_num = 0, 0, 0, 0, 0
with torch.no_grad():
    for minibatch_id in range(minibatch_num):
        minibatch = dataloader[minibatch_id]
        article = minibatch['article'].to(device)
        label = minibatch['label'].to('cpu').numpy()
        predict = model(article).to('cpu')
        predict = torch.softmax(predict, dim=1).numpy()
        predict = (predict[:, 1] > 0.5) + 0
        error = (predict != label) + 0

        # for i in range(error.shape[0]):
        #     if error[i] == 1:
        #         print(f"label = {label[i]}")
        #         print(f"predict = {predict[i]}")
        #         print(f"filename = {minibatch['filename'][i]}")
        #         print(f"row_id = {minibatch['row_id'][i]}")
        #         print(f"keywords = {minibatch['keywords'][i]}")
        #         text_article = [voc.index2word[n.item()] for n in minibatch['article'][:,i]]
        #         print(f"article = {''.join(text_article)}")
        #         print("\n\n")

        error_pos = error * (label == 1)
        error_neg = error * (label == 0)

        error_count += np.sum(error)
        error_pos_count += np.sum(error_pos)
        pos_num += sum((label == 1) + 0)
        error_neg_count += np.sum(error_neg)
        neg_num += sum((label == 0) + 0)

        error_rate = error_count / (pos_num + neg_num)
        error_pos_rate = error_pos_count / pos_num
        error_neg_rate = error_neg_count / neg_num
        print("minibatch:%5d/%d, ERROR_RATE:%10.8f, POS_ERROR_RATE:%10.8f, NEG_ERROR_RATE:%10.8f" % (
            minibatch_id, minibatch_num, error_rate, error_pos_rate, error_neg_rate))
