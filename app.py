import io
import json
import pickle
import torch
import pyltp
import tools
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # get the GPU device

##########################################################################
## load vocabulary
vocabulary_file = open(r"/home/zhangyong/Workspace/workspace3/data/vocabulary.voc", "rb")  # open the vocabulary file
voc = pickle.load(vocabulary_file)  # 载入词汇表
vocabulary_file.close() # close the vocabulary file

##########################################################################
## load the AI model
model_file = open(r"/home/zhangyong/Workspace/workspace3/model/model004.pkl", "rb")  # open the model file
model = pickle.load(model_file)  # load the model file
model.to(device)  # push model to GPU device
model.eval()  # set the model to evaluation mode, (the dropout layer need this)
model_file.close()  # close the model file

##########################################################################
## load segmentor
model_path = r"/home/zhangyong/Workspace/workspace3/ltp_data_v3.4.0/cws.model"
user_dict = r"/home/zhangyong/Workspace/workspace3/ltp_data_v3.4.0/userdict.txt"
segmentor = pyltp.Segmentor()
segmentor.load_with_lexicon(model_path, user_dict)

##########################################################################
## preprocess to normalize the html content
def preprocess(article): # preprocess the raw article represented as a html file
    article = tools.removeHTMLtag(article)  # 去除文本中的HTML标签
    article = tools.removeSpecialChar(article)  # 去除一些特殊字符
    info = tools.extractTextInfo(article)  # 从文本中抽取结构化信息
    if(tools.isContainAttachment(info['body'])):
        return False, "Can not process an article contains attachment, such as images."

    info['body'] = list(segmentor.segment(info['body']))  # 分词
    info['body'] = tools.segmentRefine_PuctuationMark(info['body'])  # 对标点符号进行更多的处理
    info['body'] = tools.segmentRefine_ChineseDigit(info['body'])  # 对中文数字进行更多的处理
    info['body'] = tools.segmentRefine_Digit(info['body'])  # 对数字进行更多的处理
    info['body'] = tools.segmentRefine_English(info['body'])  # 对英文进行更多的处理
    info['body'] = ['SOS'] + [x for x in info['body'] if x != ""] + ['EOS']
    return info

##########################################################################
## get the prediction result and score value.
## the input parameter article is a list of chinese word (the result of chinese words segmentation).
def get_prediction(article):
    # testset = tools.splitUncommonWord(testset, voc, 10)  # 如果一个多字词的出现次数少于10次，将它拆分为单字词

    article_digit = []  # we want to map the article to digit list and stored here
    for word in article:
        if word in voc.word2index:
            article_digit.append(voc.word2index[word])
        else:
            article_digit.append(voc.word2index["UNKNOWN"])

    # transform article_digit to tensor

    predict_label = 1
    predict_score = [('一个',1.1), ('is',2.1), ('a',3.1), ('pig',4.1)]
    return predict_label, predict_score

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        article = file.read() # read the article, it is stored as an html file
        #

        predict_label, predict_score = get_prediction(article=article)
        return jsonify({"predict_label" : predict_label, 'predict_score' : predict_score})

if __name__ == "__main__":
    text = """标题：发展壮大村集体经济 开启乡村振兴新征程<br/>主题：<div align="left">&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; <font size="5">壮大农村集体经济，是增强基层党组织创造力、凝聚力和战斗力的现实需要，不断壮大村集体经济是新时代的要求也是踏上新征程的有力保障。凤城市弟兄山镇结合当地实际，解放思想，真抓实干，不断探索集体经济模式，走出了发展壮大农村集体经济的广阔天地。</font></div><div align="left">&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; <font size="5"><strong>聚焦问题补齐短板，解放思想破解难题。</strong>弟兄山位于凤城北部，是一个具有矿产资源的工业小镇，农业相对落后。过去，该镇8个行政村，均没有集体收入来源，属于典型的空壳村。按照上级要求，三年全部脱壳难度极大。为切实改变这一现状，该镇党政领导敢于直面问题，解放制约村集体发展的旧思想，大胆探索，多次到东港市及凤城南部乡镇实地考察，通过集思广益、分析研判、反复论证后，确定以反季节蓝莓种植项目带动村集体经济壮大发展，并采取“飞地”模式“抱团”集中经营发展。运用“飞地经济”模式降低了发展农村集体经济的自然风险和市场风险，有效破解了村集体经济发展难题。</font></div><div align="left">&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; <font size="5"><strong>强化党建引领作用，压实压紧工作责任。</strong>将村集体经济工作列入年度基层党建专项述职评议考核的重要内容，通过召开专题研讨会，研究壮大村集体经济可行性方案，解决突出问题，强化组织领导，明确工作责任。同时，组建发展壮大村集体经济领导小组，负责指导各村发展村集体经济，统筹经管、财政、农业等成员单位，从前期设计、投审、招投标至中期工程质量监督、水电路基础设施配套，到后期大棚招租、工程结算等环节全面严格把关，落实工作任务，加强沟通协调，形成工作合力，为实现“清零”目标提供有力的组织保障。</font></div><div align="left">&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; <font size="5"><strong>积极向上争取资金，全面提升发展动力。</strong>该镇把争取项目资金支持作为发展壮大村集体经济的原动力，用项目资金撬动村级集体经济发展。2018年向上级争取到4个村扶持资金200万元，选择在东兴村一块土壤肥沃、交通便利的土地上，联合建设标准化大棚12个，总占地60亩，作为固定资产投资。引进鑫盛蓝莓公司以经营资金投入和技术作为投资合作，通过上纳租金承包大棚的形式进行生产经营，第一年4个村均已得到5万元的租金，实现了脱壳。2019年，又争取到2个村扶持资金75万，计划继续以此模式建设大棚，今年年底可顺利消除6个空壳村。</font></div><div align="left">&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; <font size="5"><strong>乡村振兴指日可待，美好生活未来可期。</strong>按照项目规划，将温室大棚建设向集群、规模化方向发展，实施后将会产生广泛的带动效应。以蓝莓种植园为中心，带动周边各村100余个已建大棚统一规范化种植。同时，该镇将分期建设种植园，反季节蓝莓种植大棚将逐步达到150个左右，初步形成规模。除可以壮大集体经济外，还将解决周边群众百余人就业。至此，该镇将告别单一传统种植业，开启乡村振兴新征程，力争走出一条新时代弟兄山特色农业现代化新路。</font></div><br/> 发稿人信息：<br/>姓名：赫星硕<br/>单位：弟兄山镇<br />邮箱：hxs0415@163.com<br />手机号：15714157111"""
    info = preprocess(text)
    print(info)