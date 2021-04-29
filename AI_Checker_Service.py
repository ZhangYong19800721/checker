# -*- coding: UTF-8 -*-

from sys import argv
import io
import json
import pickle
import torch
import pyltp
import tools
import re
import numpy as np
from flask import Flask, jsonify, request

script, service_path, ip, port = argv

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # get the GPU device

##########################################################################
## load vocabulary
vocabulary_file = open(service_path + "/GCDYW_article_AI_checker/data/vocabulary.voc", "rb")  # open the vocabulary file
voc = pickle.load(vocabulary_file)  # 载入词汇表
vocabulary_file.close()  # close the vocabulary file

##########################################################################
## load the AI model
model_file = open(service_path + "/GCDYW_article_AI_checker/model/model004.pkl", "rb")  # open the model file
model = pickle.load(model_file)  # load the model file
model.to(device)  # push model to GPU device
model.eval()  # set the model to evaluation mode, (the dropout layer need this)
model_file.close()  # close the model file

##########################################################################
## load segmentor
model_path = service_path + "/GCDYW_article_AI_checker/ltp_data_v3.4.0/cws.model"
user_dict = service_path + "/GCDYW_article_AI_checker/ltp_data_v3.4.0/userdict.txt"
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(model_path, user_dict)


##########################################################################
## 将article从未在词表中出现的多字词切分为单字词
def splitUnknownWord(article, voc):
    wordList = []
    for word in article:
        if word in voc.word2count:
            wordList.append(word)
        elif len(word) > 1:
            L = re.split(r"([\W\w])", word)
            L = [x for x in L if x != ""]
            for c in L:
                if c in voc.word2count:
                    wordList.append(c)
                else:
                    wordList.append("UNKNOWN")
        else:
            wordList.append("UNKNOWN")
    article = [x for x in wordList if x != ""]
    return article


##########################################################################
## preprocess to normalize the html content
def preprocess(article):  # preprocess the raw article represented as a html file
    article = tools.removeHTMLtag(article)  # 去除文本中的HTML标签
    article = tools.removeSpecialChar(article)  # 去除一些特殊字符
    info = tools.extractTextInfo(article)  # 从文本中抽取结构化信息
    if info == None:
        return False, "Can not extract architecture infomation from article"

    if (tools.isContainAttachment(info['body'])):
        return False, "Can not process an article contains attachment, such as images."

    info['body'] = list(segmentor.segment(info['body']))  # 分词
    info['body'] = tools.segmentRefine_PuctuationMark(info['body'])  # 对标点符号进行更多的处理
    info['body'] = tools.segmentRefine_ChineseDigit(info['body'])  # 对中文数字进行更多的处理
    info['body'] = tools.segmentRefine_Digit(info['body'])  # 对数字进行更多的处理
    info['body'] = tools.segmentRefine_English(info['body'])  # 对英文进行更多的处理
    info['body'] = ['SOS'] + [x for x in info['body'] if x != ""] + ['EOS']
    info['body'] = splitUnknownWord(info['body'], voc)
    return True, info


##########################################################################
## postProcess to align to the original html content
def postProcess(original_article, words_score):
    words_score.pop(-1)  # remove the EOS token
    words_score.pop(0)  # remove the SOS token
    clean_article, score = "", []
    for x, y in words_score:
        clean_article += x
        score += [y] * len(x)

    match_idx = Needleman_Wunsch(original_article, clean_article)
    # DEBUG ###########################################################
    # debug_match_idx = match_idx
    # for i in range(len(original_article)):
    #     print(original_article[i], "--->",
    #           debug_match_idx[i], "--->",
    #           clean_article[debug_match_idx[i]] if debug_match_idx[i] != -1 else "", "--->",
    #           score[debug_match_idx[i]] if debug_match_idx[i] != -1 else "")
    ###################################################################
    color_start, color_end = "<font color=#40a944>", "</font>"
    color_start, color_end = color_start[::-1], color_end[::-1]
    reverse_colored_article = ""
    idx = len(match_idx) - 1
    while idx >= 0:
        if match_idx[idx] == -1:
            reverse_colored_article += original_article[idx]
        elif score[match_idx[idx]] >= 8:
            reverse_colored_article += color_end + original_article[idx] + color_start
        else:
            reverse_colored_article += original_article[idx]
        idx -= 1
    colored_article = reverse_colored_article[::-1]
    return colored_article


###########################################################################
# Use Needleman Wunsch algotithm to align string A and string B
def Needleman_Wunsch(strA, strB):
    strlenA = len(strA)
    strlenB = len(strB)
    score_table = np.zeros((strlenA + 1, strlenB + 1), dtype=np.long)
    for n in range(1, strlenA + 1):
        score_table[n][0] = score_table[n - 1][0] - 3
    for n in range(1, strlenB + 1):
        score_table[0][n] = score_table[0][n - 1] - 3

    for r in range(1, strlenA + 1):
        for c in range(1, strlenB + 1):
            F1 = score_table[r - 1][c - 1]
            F1 = F1 + (8 if strA[r - 1] == strB[c - 1] else -5)
            F2 = score_table[r - 1][c] - 3
            F3 = score_table[r][c - 1] - 3
            score_table[r][c] = max(F1, F2, F3)

    idxA, idxB, result = strlenA, strlenB, []
    while idxA > 0 or idxB > 0:
        if idxA != 0 and idxB != 0:
            R0 = score_table[idxA][idxB]
            R1 = score_table[idxA - 1][idxB - 1]
            R2 = score_table[idxA][idxB - 1]
            R3 = score_table[idxA - 1][idxB]
            if R1 >= max(R2, R3):
                if R0 >= R1:
                    result += [idxB-1]  # match
                else:
                    result += [-1]      # dismatch
                idxA, idxB = idxA - 1, idxB - 1
            elif R2 >= max(R1, R3):
                idxA, idxB = idxA, idxB - 1
            elif R3 >= max(R1, R2):
                result += [-1]
                idxA, idxB = idxA - 1, idxB
        elif idxB == 0:
            result += [-1]
            idxA = idxA - 1

    result.reverse()
    return result


##########################################################################
## get the prediction result and score value.
## the input parameter article is a list of chinese word (the result of chinese words segmentation).
def get_prediction(article):
    article_digit = []  # we want to map the article to digit list and stored here
    for word in article:
        if word in voc.word2index:
            article_digit.append(voc.word2index[word])
        else:
            article_digit.append(voc.word2index["UNKNOWN"])

    # transform article_digit to tensor
    article_len = len(article_digit)
    # print('article_len = ', article_len)
    article_digit = tools.zeroPadding([article_digit, article_digit])
    article_tensor = torch.LongTensor(article_digit)  # 将列表转换为张量
    article_tensor = article_tensor.to(device)

    with torch.no_grad():
        predict_label, predict_score = model(article_tensor, [article_len, article_len])
        predict_label, predict_score = predict_label.to('cpu'), predict_score.to('cpu')
        predict_label = torch.softmax(predict_label, dim=1).numpy()
        predict_label = (predict_label[:, 1] > 0.5) + 0

    predict_label = int(predict_label[0])  # change the int64 to int32, or else can not be jsonify.
    predict_score = predict_score[:, 0].numpy()  # this is a tensor, we need to change it to numpy array
    average_score = 1 / article_len
    score = []
    for i in range(article_len):
        if predict_score[i] <= 1 * average_score:
            score.append((voc.index2word[article_digit[i][0]], 1))
        elif predict_score[i] <= 2 * average_score:
            score.append((voc.index2word[article_digit[i][0]], 2))
        elif predict_score[i] <= 3 * average_score:
            score.append((voc.index2word[article_digit[i][0]], 3))
        elif predict_score[i] <= 4 * average_score:
            score.append((voc.index2word[article_digit[i][0]], 4))
        elif predict_score[i] <= 5 * average_score:
            score.append((voc.index2word[article_digit[i][0]], 5))
        elif predict_score[i] <= 6 * average_score:
            score.append((voc.index2word[article_digit[i][0]], 6))
        elif predict_score[i] <= 7 * average_score:
            score.append((voc.index2word[article_digit[i][0]], 7))
        else:
            score.append((voc.index2word[article_digit[i][0]], 8))
    # print("score = ", score)
    return predict_label, score


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        original_article = request.args.get('article')
        result, article = preprocess(original_article)
        if result == True:
            predict_label, predict_score = get_prediction(article=article['body'])
            colored_article = postProcess(original_article, predict_score) if predict_label == 0 else original_article
            # return json.dumps({"predict_label": predict_label, "colored_article": colored_article, "predict_score": predict_score})
            return json.dumps({"predict_label": predict_label, "colored_article": colored_article})
        else:
            return json.dumps({"error": article})


if __name__ == "__main__":
    # original_article = """标题：牢记初心砥砺奋进<br/>主题：<div align="left"><font face="方正仿宋_GBK"><font color="#000000">习近平总书记在</font></font><font color="#000000">“不忘初心、牢记使命”主题教育工作会议上的重要讲话让我深受启发，我认为一名合格的共产党员务必做到以下三点。</font></div><div align="left"><font face="方正仿宋_GBK"><font color="#000000">一是加强理论学习。理论学习有收获，重点是教育引导广大党员干部在原有学习的基础上取得新进步，加深对新时代中国特色社会主义思想和党中央大政方针的理解，学深悟透、融会贯通，增强贯彻落实的自觉性和坚定性，提高运用党的创新理论指导实践、推动工作的能力。要充分认识加强理论学习、提高理论水平的重要性，自觉加强理论学习，切实提高执政能力，进一步增强党的领导核心作用。党的理论创新是永无止境的，这就要求广大党员干部要紧跟党的理论创新步伐，对党提出的新思想新理论要常学常新。要统筹好工学关系，把理论学习当成做好工作的必要环节，切实用党的最新思想理论指明工作努力方向。要坚持学习原文，深入研读党的最新理论著作，原滋原味地领悟思想的魅力、理论的力量。要坚持研讨交流，通过与他人交流讨论碰撞思维，更加多层次、多角度地理解党的思想理论。</font></font></div><div align="left"><font face="方正仿宋_GBK"><font color="#000000">二是接受</font></font><font face="方正仿宋_GBK">思想政治洗礼，引导广大党员干部坚定对马克思主义的信仰、对中国特色社会主义的信念，传承红色基因，增强</font>“四个意识”、坚定“四个自信”、做到“两个维护”，自觉在思想上政治上行动上同党中央保持高度一致，始终忠诚于党、忠诚于人民、忠诚于马克思主义。<font face="方正仿宋_GBK"><font color="#000000">学而不思则罔，通过思考增强对理论的内心认同是至关重要的一环。要结合自身勤思考，查找自身在思想境界上存在的不足，在理论认识上存在的误区，切实补好精神之</font></font><font color="#000000">“钙”、纠正认识之“偏”。要结合工作勤思考，始终坚持把讲政治放在第一位，始终践行以人民为中心的理念，坚定对党忠诚、为民服务的思想自觉。要立足发展勤思考，按照党的最新理论和奋斗目标，调准人生奋斗的航标、工作努力的方向，始终沿着正确的道路前行。</font></div><div align="left"><font face="方正仿宋_GBK"><font color="#000000">三是要把理论知识落到实处。</font></font><font face="方正仿宋_GBK">结合实际，创造性开展工作，把学习教育、调查研究、检视问题、整改落实贯穿主题教育全过程，努力取得最好成效。</font><font face="方正仿宋_GBK"><font color="#000000">要勤于践行。不干，半点马克思主义都没有。学习理论最终是指导实践，关键是知行合一。要在重大考验中践行，增强政治定力、政治判断力和政治执行力，切实树牢</font></font><font color="#000000">“四个意识”，坚定“四个自信”。要在工作履职中践行，坚决贯彻执行党的方针政策和改革举措，不忘初心、牢记使命，为实现人民群众对美好生活的向往而努力奋斗。要在日常生活中践行，牢记党员身份，从平时做起，从小事做起，做一个讲社会公德、职业道德、家庭美德和个人品德的党员干部。</font></div><br /><br/> 发稿人信息：<br/>姓名：桂光馨<br/>单位：重庆市开州区文峰街道办事处<br />邮箱：624609245@qq.com<br />手机号：17723235348"""
    # article = preprocess(original_article)
    # predict_label, predict_score = get_prediction(article=article['body'])
    # colored_article = postProcess(original_article, predict_score)
    # print(predict_label)
    # print(colored_article)
    app.run(host=ip, port=int(port), debug='True')
