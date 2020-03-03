# coding=utf-8

import pickle
import DATASET
import torch
import tools
import numpy as np

# 读入词汇表
vocabulary_file = open(r"./data/vocabulary.voc", "rb")
voc = pickle.load(vocabulary_file)  # 载入词汇表
vocabulary_file.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_file = open(r"./model/model004.pkl", "rb")
model = pickle.load(model_file)
model.to(device)
model.eval()
model_file.close()

testlog_file = open("testlog.txt", "w", encoding='utf-8')

testset = DATASET.GCDYW(r"./data/testset_digit.cps")  # load the test data
testset.trim(20, 1000)
minibatch_size = 120
dataloader = DATASET.TEST_LOADER(testset, minibatch_size=minibatch_size)  # set the minibatch size
minibatch_num = len(dataloader)

error_count, error_pos_count, error_neg_count, pos_num, neg_num = 0, 0, 0, 0, 0
with torch.no_grad():
    for minibatch_id in range(minibatch_num):
        minibatch = dataloader[minibatch_id]
        article = minibatch['article'].to(device)
        article_len = minibatch['article_len']
        label = minibatch['label'].to('cpu').numpy()
        predict, score = model(article, article_len)
        predict, score = predict.to('cpu'), score.to('cpu')
        predict = torch.softmax(predict, dim=1).numpy()
        predict = (predict[:, 1] > 0.5) + 0
        error = (predict != label) + 0

        for i in range(error.shape[0]):
            if label[i] == 0 or (error[i] == 1 and label[i] == 1):
                print(f"label = {label[i]}")
                print(f"predict = {predict[i]}")
                print(f"filename = {minibatch['filename'][i]}")
                print(f"row_id = {minibatch['row_id'][i]}")
                print(f"keywords = {minibatch['keywords'][i]}")
                text_article = [voc.index2word[n.item()] for n in minibatch['article'][:, i]]
                text_article_prob = [(text_article[n], score[n][i].item()) for n in range(len(text_article))]
                tools.printColorAriticle(text_article_prob)
                print()

        for i in range(error.shape[0]):
            if error[i] == 1:
                testlog_file.write(f"label = {label[i]}\n")
                testlog_file.write(f"predict = {predict[i]}\n")
                testlog_file.write(f"filename = {minibatch['filename'][i]}\n")
                testlog_file.write(f"row_id = {minibatch['row_id'][i]}\n")
                testlog_file.write(f"keywords = {minibatch['keywords'][i]}\n")
                text_article = [voc.index2word[n.item()] for n in minibatch['article'][:,i]]
                text_article_prob = [(text_article[n], score[n][i].item()) for n in range(len(text_article))]
                # testlog_file.write(f"article = {''.join(text_article)}\n")
                testlog_file.write(f"article = {text_article_prob}\n")
                testlog_file.write("\n\n")
                testlog_file.flush()

        error_pos = error * (label == 1)
        error_neg = error * (label == 0)

        error_count += np.sum(error)
        error_pos_count += np.sum(error_pos)
        pos_num += sum((label == 1) + 0)
        error_neg_count += np.sum(error_neg)
        neg_num += sum((label == 0) + 0)

        error_rate = error_count / (pos_num + neg_num)
        pos_recall = (pos_num - error_pos_count) / (pos_num - error_pos_count + error_neg_count) if pos_num - error_pos_count + error_neg_count != 0 else 0
        neg_recall = (neg_num - error_neg_count) / (neg_num - error_neg_count + error_pos_count) if neg_num - error_neg_count + error_pos_count != 0 else 0
        error_pos_rate = error_pos_count / pos_num if pos_num != 0 else 0
        error_neg_rate = error_neg_count / neg_num if neg_num != 0 else 0
        print("minibatch:%5d/%d, ERROR_RATE:%10.8f, POS_RECALL:%10.8f, NEG_RECALL:%10.8f, POS_ERROR_RATE:%10.8f, NEG_ERROR_RATE:%10.8f, POSNUM:%06d, NEGNUM:%06d" % (
            minibatch_id, minibatch_num, error_rate, pos_recall, neg_recall, error_pos_rate, error_neg_rate, pos_num, neg_num))

testlog_file.close()