# coding=utf-8

import pickle
import DATASET
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_file = open(r"./model/model_final.pkl", "rb")
model = pickle.load(model_file)
model.to(device)
model.eval()
model_file.close()

testset = DATASET.GCDYW(r"./data/corpus_testset_consistent_rmrepeat_digit.cps")  # load the test data
testset.trim(20,1000)
minibatch_size = 25
dataloader = DATASET.LOADER(testset, minibatch_size=minibatch_size)  # set the minibatch size
minibatch_num = len(dataloader)

error_count, error_pos_count, error_neg_count, pos_num, neg_num = 0, 0, 0, 0, 0
with torch.no_grad():
    for minibatch_id in range(minibatch_num):
        minibatch = dataloader[minibatch_id]
        article = minibatch['article'].to(device)
        label = minibatch['label'].numpy()
        predict = model(article).to('cpu').numpy()
        predict = np.argmax(predict, axis=1)

        error = (predict != label) + 0
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
        print("minibatch:%5d/%d, 错误率:%10.8f, 正错误率:%10.8f, 反错误率:%10.8f" % (
        minibatch_id, minibatch_num, error_rate, error_pos_rate, error_neg_rate))
