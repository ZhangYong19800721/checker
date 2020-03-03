# coding=utf-8
import re
import xlrd  # 读取excel文件
import itertools
import torch  # 导入torch模块
import torch.nn as nn  # 导入torch的神经网络模块
from torch import optim  # 导入torch的最优化模块
import torch.nn.functional as F  # 导入torch神经网络子模块的函数模块

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# 判断文本中是否包含附件
def isContainAttachment(text):
    flag1 = text.find("附件：") != -1
    flag2 = text.find("附件:") != -1
    flag3 = text.find(".jpg") != -1
    flag4 = text.find(".JPG") != -1
    flag5 = text.find(".jpeg") != -1
    flag6 = text.find(".JPEG") != -1
    flag7 = text.find("[attach]") != -1
    return flag1 or flag2 or flag3 or flag4 or flag5 or flag6 or flag7


# 去除文本中的HTML标签
def removeHTMLtag(text):
    new_text = text
    new_text = re.sub(r"<br\s*/>", "\n", new_text)  # <br/>标签替换为换行
    new_text = re.sub(r"</*div[^>]+>", " ", new_text)  # <div>和</div>标签替换为空格
    new_text = re.sub(r"&nbsp;", " ", new_text)  # &nbsp; 标签替换为空格
    new_text = re.sub(r"&quot;", "\"", new_text)  # &quot; 标签替换为英文的双引号
    # new_text = re.sub(r"\xa0", " ", new_text)  # \xa0 替换为空格
    new_text = re.sub(r"<[^>]+>", "", new_text)  # 其他HTML标签替换为空字符串
    return new_text


# http://tool.sufeinet.com/Code/ChineseUnicode.aspx?t=2&str=%5Cu2022 通过该网址可以查看特殊unicode字符表示的符号
def removeSpecialChar(text):
    new_text = text

    new_text = re.sub(r"·", ".", new_text)
    new_text = re.sub(r"\u0001", " ", new_text)
    new_text = re.sub(r"\u0002", " ", new_text)
    new_text = re.sub(r"\u0004", " ", new_text)
    new_text = re.sub(r"\u0008", " ", new_text)
    new_text = re.sub(r"\u000b", " ", new_text)
    new_text = re.sub(r"\u0010", " ", new_text)
    new_text = re.sub(r"\u001f", " ", new_text)
    new_text = re.sub(r"\u003f", "?", new_text)  #
    new_text = re.sub(r"\u007f", " ", new_text)
    new_text = re.sub(r"\u009f", " ", new_text)
    new_text = re.sub(r"\u02d9", ".", new_text)
    new_text = re.sub(r"\u1173", "一", new_text)  #
    new_text = re.sub(r"\u2002", " ", new_text)  #
    new_text = re.sub(r"\u2003", " ", new_text)  #
    new_text = re.sub(r"\u200b", " ", new_text)  #
    new_text = re.sub(r"\u200d", " ", new_text)  #
    new_text = re.sub(r"\u200e", " ", new_text)  #
    new_text = re.sub(r"\u2018", "'", new_text)
    new_text = re.sub(r"\u201a", "，", new_text)  #
    new_text = re.sub(r"\u201b", "‘", new_text)  #
    new_text = re.sub(r"\u201e", "\"", new_text)
    new_text = re.sub(r"\u2022", ".", new_text)  #
    new_text = re.sub(r"\u203a", ">", new_text)  #
    new_text = re.sub(r"\u2160", "(1)", new_text)  #
    new_text = re.sub(r"\u2161", "(2)", new_text)
    new_text = re.sub(r"\u2162", "(3)", new_text)
    new_text = re.sub(r"\u2163", "(4)", new_text)
    new_text = re.sub(r"\u2164", "(5)", new_text)
    new_text = re.sub(r"\u2165", "(6)", new_text)
    new_text = re.sub(r"\u2166", "(7)", new_text)
    new_text = re.sub(r"\u2167", "(8)", new_text)
    new_text = re.sub(r"\u2168", "(9)", new_text)
    new_text = re.sub(r"\u2169", "(10)", new_text)
    new_text = re.sub(r"\u216a", "(11)", new_text)
    new_text = re.sub(r"\u216b", "(12)", new_text)
    new_text = re.sub(r"\u2215", "/", new_text)  #
    new_text = re.sub(r"\u2219", ".", new_text)  #
    new_text = re.sub(r"\u2236", ":", new_text)  #
    new_text = re.sub(r"\u2460", "(1)", new_text)  #
    new_text = re.sub(r"\u2461", "(2)", new_text)  #
    new_text = re.sub(r"\u2462", "(3)", new_text)  #
    new_text = re.sub(r"\u2463", "(4)", new_text)  #
    new_text = re.sub(r"\u2464", "(5)", new_text)  #
    new_text = re.sub(r"\u2465", "(6)", new_text)  #
    new_text = re.sub(r"\u2466", "(7)", new_text)  #
    new_text = re.sub(r"\u2467", "(8)", new_text)  #
    new_text = re.sub(r"\u2468", "(9)", new_text)  #
    new_text = re.sub(r"\u2469", "(10)", new_text)  #
    new_text = re.sub(r"\u246a", "(11)", new_text)  #
    new_text = re.sub(r"\u246b", "(12)", new_text)  #
    new_text = re.sub(r"\u246c", "(13)", new_text)  #
    new_text = re.sub(r"\u246d", "(14)", new_text)  #
    new_text = re.sub(r"\u246e", "(15)", new_text)  #
    new_text = re.sub(r"\u246f", "(16)", new_text)  #
    new_text = re.sub(r"\u2470", "(17)", new_text)  #
    new_text = re.sub(r"\u2471", "(18)", new_text)  #
    new_text = re.sub(r"\u2472", "(19)", new_text)  #
    new_text = re.sub(r"\u2473", "(20)", new_text)  #
    new_text = re.sub(r"\u2474", "(1)", new_text)
    new_text = re.sub(r"\u2475", "(2)", new_text)
    new_text = re.sub(r"\u2476", "(3)", new_text)
    new_text = re.sub(r"\u2477", "(4)", new_text)
    new_text = re.sub(r"\u2478", "(5)", new_text)
    new_text = re.sub(r"\u2479", "(6)", new_text)
    new_text = re.sub(r"\u247a", "(7)", new_text)
    new_text = re.sub(r"\u247b", "(8)", new_text)
    new_text = re.sub(r"\u247c", "(9)", new_text)
    new_text = re.sub(r"\u247d", "(10)", new_text)
    new_text = re.sub(r"\u247e", "(11)", new_text)
    new_text = re.sub(r"\u247f", "(12)", new_text)
    new_text = re.sub(r"\u2480", "(13)", new_text)
    new_text = re.sub(r"\u2481", "(14)", new_text)
    new_text = re.sub(r"\u2482", "(15)", new_text)
    new_text = re.sub(r"\u2483", "(16)", new_text)
    new_text = re.sub(r"\u2484", "(17)", new_text)
    new_text = re.sub(r"\u2485", "(18)", new_text)
    new_text = re.sub(r"\u2486", "(19)", new_text)
    new_text = re.sub(r"\u2487", "(20)", new_text)
    new_text = re.sub(r"\u2488", "(1)", new_text)
    new_text = re.sub(r"\u2489", "(2)", new_text)
    new_text = re.sub(r"\u248a", "(3)", new_text)
    new_text = re.sub(r"\u248b", "(4)", new_text)
    new_text = re.sub(r"\u248c", "(5)", new_text)
    new_text = re.sub(r"\u248d", "(6)", new_text)
    new_text = re.sub(r"\u248e", "(7)", new_text)
    new_text = re.sub(r"\u248f", "(8)", new_text)
    new_text = re.sub(r"\u2490", "(9)", new_text)
    new_text = re.sub(r"\u2491", "(10)", new_text)
    new_text = re.sub(r"\u2492", "(11)", new_text)
    new_text = re.sub(r"\u2493", "(12)", new_text)
    new_text = re.sub(r"\u2494", "(13)", new_text)
    new_text = re.sub(r"\u2495", "(14)", new_text)
    new_text = re.sub(r"\u2496", "(15)", new_text)
    new_text = re.sub(r"\u2497", "(16)", new_text)
    new_text = re.sub(r"\u2498", "(17)", new_text)
    new_text = re.sub(r"\u2499", "(18)", new_text)
    new_text = re.sub(r"\u249a", "(19)", new_text)
    new_text = re.sub(r"\u249b", "(20)", new_text)
    new_text = re.sub(r"\u2500", "—", new_text)
    new_text = re.sub(r"\u2776", "(1)", new_text)
    new_text = re.sub(r"\u2777", "(2)", new_text)
    new_text = re.sub(r"\u2778", "(3)", new_text)
    new_text = re.sub(r"\u2779", "(4)", new_text)
    new_text = re.sub(r"\u277a", "(5)", new_text)
    new_text = re.sub(r"\u277b", "(6)", new_text)
    new_text = re.sub(r"\u277c", "(7)", new_text)
    new_text = re.sub(r"\u277d", "(8)", new_text)
    new_text = re.sub(r"\u277e", "(9)", new_text)
    new_text = re.sub(r"\u277f", "(10)", new_text)
    new_text = re.sub(r"\u2f00", "一", new_text)
    new_text = re.sub(r"\u2f08", "人", new_text)  #
    new_text = re.sub(r"\u2f09", "儿", new_text)
    new_text = re.sub(r"\u2f42", "文", new_text)  #
    new_text = re.sub(r"\u2f46", "无", new_text)
    new_text = re.sub(r"\u2f54", "水", new_text)
    new_text = re.sub(r"\u2f63", "生", new_text)
    new_text = re.sub(r"\u2f84", "至", new_text)
    new_text = re.sub(r"\u2f9b", "走", new_text)  #
    new_text = re.sub(r"\u30aa", "才", new_text)  #
    new_text = re.sub(r"\u25aa", ".", new_text)  #
    new_text = re.sub(r"\u25cb", "零", new_text)
    new_text = re.sub(r"\u2f00", "一", new_text)  #
    new_text = re.sub(r"\u2fbc", "高", new_text)
    new_text = re.sub(r"\u3000", " ", new_text)  # 全角空格
    new_text = re.sub(r"\u3007", "零", new_text)
    new_text = re.sub(r"\u301c", "~", new_text)
    new_text = re.sub(r"\u30cb", "二", new_text)
    new_text = re.sub(r"\u30fb", "*", new_text)
    new_text = re.sub(r"\u30fc", "一", new_text)  #
    new_text = re.sub(r"\u3220", "(1)", new_text)
    new_text = re.sub(r"\u3221", "(2)", new_text)
    new_text = re.sub(r"\u3222", "(3)", new_text)
    new_text = re.sub(r"\u3223", "(4)", new_text)
    new_text = re.sub(r"\u3224", "(5)", new_text)
    new_text = re.sub(r"\u3225", "(6)", new_text)
    new_text = re.sub(r"\u3226", "(7)", new_text)
    new_text = re.sub(r"\u3227", "(8)", new_text)
    new_text = re.sub(r"\u3228", "(9)", new_text)
    new_text = re.sub(r"\u3229", "(10)", new_text)
    new_text = re.sub(r"\u3251", "(21)", new_text)  #
    new_text = re.sub(r"\u3252", "(22)", new_text)  #
    new_text = re.sub(r"\u3253", "(23)", new_text)  #
    new_text = re.sub(r"\u3254", "(24)", new_text)  #
    new_text = re.sub(r"\u3255", "(25)", new_text)  #
    new_text = re.sub(r"\u3256", "(26)", new_text)  #
    new_text = re.sub(r"\u3257", "(27)", new_text)  #
    new_text = re.sub(r"\u3258", "(28)", new_text)  #
    new_text = re.sub(r"\u3259", "(29)", new_text)  #
    new_text = re.sub(r"\ue001", " ", new_text)  #
    new_text = re.sub(r"\ue004", " ", new_text)
    new_text = re.sub(r"\ue007", " ", new_text)
    new_text = re.sub(r"\ue5cf", " ", new_text)
    new_text = re.sub(r"\ue600", " ", new_text)
    new_text = re.sub(r"\ue737", " ", new_text)
    new_text = re.sub(r"\ufe50", "，", new_text)
    new_text = re.sub(r"\ufe51", "、", new_text)
    new_text = re.sub(r"\ufe52", ".", new_text)
    new_text = re.sub(r"\ufe54", ";", new_text)
    new_text = re.sub(r"\ufe62", "+", new_text)
    new_text = re.sub(r"\ufe6a", "%", new_text)
    new_text = re.sub(r"\uff01", "!", new_text)
    new_text = re.sub(r"\uff02", "\"", new_text)
    new_text = re.sub(r"\uff05", "%", new_text)  #
    new_text = re.sub(r"\uff08", "(", new_text)
    new_text = re.sub(r"\uff09", ")", new_text)
    new_text = re.sub(r"\uff0b", "+", new_text)
    new_text = re.sub(r"\uff0d", "-", new_text)
    new_text = re.sub(r"\uff0e", ".", new_text)
    new_text = re.sub(r"\uff10", "0", new_text)  #
    new_text = re.sub(r"\uff11", "1", new_text)
    new_text = re.sub(r"\uff12", "2", new_text)
    new_text = re.sub(r"\uff13", "3", new_text)
    new_text = re.sub(r"\uff14", "4", new_text)
    new_text = re.sub(r"\uff15", "5", new_text)
    new_text = re.sub(r"\uff16", "6", new_text)
    new_text = re.sub(r"\uff17", "7", new_text)
    new_text = re.sub(r"\uff18", "8", new_text)
    new_text = re.sub(r"\uff19", "9", new_text)
    new_text = re.sub(r"\uff1a", ":", new_text)
    new_text = re.sub(r"\uff1e", ">", new_text)
    new_text = re.sub(r"\uff20", "@", new_text)
    new_text = re.sub(r"\uff21", "A", new_text)
    new_text = re.sub(r"\uff22", "B", new_text)
    new_text = re.sub(r"\uff23", "C", new_text)
    new_text = re.sub(r"\uff24", "D", new_text)
    new_text = re.sub(r"\uff25", "E", new_text)
    new_text = re.sub(r"\uff26", "F", new_text)
    new_text = re.sub(r"\uff27", "G", new_text)
    new_text = re.sub(r"\uff28", "H", new_text)
    new_text = re.sub(r"\uff29", "I", new_text)
    new_text = re.sub(r"\uff2a", "J", new_text)
    new_text = re.sub(r"\uff2b", "K", new_text)
    new_text = re.sub(r"\uff2c", "L", new_text)
    new_text = re.sub(r"\uff2d", "M", new_text)
    new_text = re.sub(r"\uff2e", "N", new_text)
    new_text = re.sub(r"\uff2f", "O", new_text)
    new_text = re.sub(r"\uff30", "P", new_text)
    new_text = re.sub(r"\uff31", "Q", new_text)
    new_text = re.sub(r"\uff32", "R", new_text)
    new_text = re.sub(r"\uff33", "S", new_text)
    new_text = re.sub(r"\uff34", "T", new_text)
    new_text = re.sub(r"\uff35", "U", new_text)
    new_text = re.sub(r"\uff36", "V", new_text)
    new_text = re.sub(r"\uff37", "W", new_text)
    new_text = re.sub(r"\uff38", "X", new_text)
    new_text = re.sub(r"\uff39", "Y", new_text)
    new_text = re.sub(r"\uff3a", "Z", new_text)
    new_text = re.sub(r"\uff41", "a", new_text)
    new_text = re.sub(r"\uff42", "b", new_text)
    new_text = re.sub(r"\uff43", "c", new_text)
    new_text = re.sub(r"\uff44", "d", new_text)
    new_text = re.sub(r"\uff45", "e", new_text)
    new_text = re.sub(r"\uff46", "f", new_text)
    new_text = re.sub(r"\uff47", "g", new_text)
    new_text = re.sub(r"\uff48", "h", new_text)
    new_text = re.sub(r"\uff49", "i", new_text)
    new_text = re.sub(r"\uff4a", "j", new_text)
    new_text = re.sub(r"\uff4b", "k", new_text)
    new_text = re.sub(r"\uff4c", "l", new_text)
    new_text = re.sub(r"\uff4d", "m", new_text)
    new_text = re.sub(r"\uff4e", "n", new_text)
    new_text = re.sub(r"\uff4f", "o", new_text)
    new_text = re.sub(r"\uff50", "p", new_text)
    new_text = re.sub(r"\uff51", "q", new_text)
    new_text = re.sub(r"\uff52", "r", new_text)
    new_text = re.sub(r"\uff53", "s", new_text)
    new_text = re.sub(r"\uff54", "t", new_text)
    new_text = re.sub(r"\uff55", "u", new_text)
    new_text = re.sub(r"\uff56", "v", new_text)
    new_text = re.sub(r"\uff57", "w", new_text)
    new_text = re.sub(r"\uff58", "x", new_text)
    new_text = re.sub(r"\uff59", "y", new_text)
    new_text = re.sub(r"\uff5a", "z", new_text)
    new_text = re.sub(r"\ufffc", " ", new_text)  #
    new_text = re.sub(r"\ufffd", " ", new_text)  #
    new_text = re.sub(r"\uff1f", "?", new_text)
    new_text = re.sub(r"。{3,}", "…", new_text)  # 将连续的3个以上句号替换为省略号
    new_text = re.sub(r"。{2}", "。", new_text)  # 将连续的2个句号替换为1个句号
    new_text = re.sub(r"\.{3,}", "…", new_text)  # 将连续的3个以上.替换为省略号
    new_text = re.sub(r"\.{2}", ".", new_text)  # 将连续的2个.替换为1个.

    return new_text


def removeSpecialStr(text):
    new_text = text
    # 替换网址
    re_url_str = r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"
    new_text = re.sub(re_url_str, "&&&&&URL&&&&&", new_text)
    # 替换邮箱
    re_str_email = r"[A-Za-z0-9_\u4e00-\u9fa5]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+"
    new_text = re.sub(re_str_email, "&&&&&EMAIL&&&&&", new_text)
    return new_text


def extractTextInfo(text):
    message = text
    contents = {}
    # 标题：...主题：...发稿人信息：...姓名：... 单位：... 邮箱：... 手机号：... 类型1的文章
    re_str_type1 = r"(标题[：:])([\w\W]*(?=主题))(主题[：:])([\w\W]*(?=发稿人信息))(发稿人信息[：:])([\w\W]*(?=姓名))(姓名[：:])([\w\W]*(?=单位))(单位[：:])([\w\W]*(?=邮箱))(邮箱[：:])([\w\W]*(?=手机号))(手机号[：:])([\w\W]*)"
    re_exp_type1 = re.search(re_str_type1, message)
    if re_exp_type1:
        contents['title'] = re_exp_type1.group(2)
        contents['body'] = contents['title'] + "。 " + re_exp_type1.group(4)
        contents['name'] = re_exp_type1.group(8)
        contents['department'] = re_exp_type1.group(10)
        contents['email'] = re_exp_type1.group(12)
        contents['mobilephone'] = re_exp_type1.group(14)
        for k, v in contents.items():
            contents[k] = re.sub(r"[\n\r]+", " ", v).strip(" ")

    # 标题：...描述：...提问人信息：... 类型2的文章
    re_str_type2 = r"(标题[：:])([\w\W]*(?=描述))(描述[：:])([\w\W]*(?=提问人信息))(提问人信息[：:])([\w\W]*)"
    re_exp_type2 = re.search(re_str_type2, message)
    if re_exp_type2:
        contents['title'] = re_exp_type2.group(2)
        contents['description'] = re_exp_type2.group(4)
        contents['name'] = re_exp_type2.group(6)
        contents['body'] = contents['title'] + "。 " + contents['description']
        for k, v in contents.items():
            contents[k] = re.sub(r"[\n\r]+", " ", v).strip(" ")

    # 回复内容：...回复附件：... 类型3的文章
    re_str_type3 = r"(回复内容[：:])([\w\W]*(?=回复附件))(回复附件[：:])([\w\W]*)"
    re_exp_type3 = re.search(re_str_type3, message)
    if re_exp_type3:
        contents['reply'] = re_exp_type3.group(2)
        contents['attach'] = re_exp_type3.group(4)
        contents['body'] = contents['reply']
        for k, v in contents.items():
            contents[k] = re.sub(r"[\n\r]+", " ", v).strip(" ")

    if re_exp_type1 or re_exp_type2 or re_exp_type3:
        contents['body'] = removeSpecialStr(contents['body'])
        return contents
    else:
        return None


# 从excel表中读取数据并进行第1步预处理，结果放在列表中返回
def readRawData(filename, rowid=None):
    result = []
    data = xlrd.open_workbook(filename)  # 获取数据
    table = data.sheet_by_name('Sheet1')  # 获取Sheet表格
    nrows = table.nrows  # 获取总行数
    ncols = table.ncols  # 获取总列数

    if not rowid:
        row_range = range(1, nrows)
    else:
        row_range = [rowid]

    for i in row_range:
        label = table.cell(i, 0).value  # 获取一个单元格的数值，第0列是标签
        keywords = table.cell(i, 1).value.encode('utf-8').decode('utf-8')  # 获取一个单元格的数值，第1列是关键词
        text = table.cell(i, 2).value.encode('utf-8').decode('utf-8')  # 获取一个单元格的数值，第2列是文本
        text = removeHTMLtag(text)  # 去除文本中的HTML标签
        text = removeSpecialChar(text)  # 去除一些特殊字符
        info = extractTextInfo(text)  # 从文本中抽取结构化信息
        if i % 1000 == 0:
            print(f"filename = {filename}, rowid = {i}")
        if info:
            if label == 2 or label == 6:
                info['label'] = 'pass'  # 标签为2或6表示审核通过
            elif label == 3 or label == 4:
                info['label'] = 'reject'  # 标签为3或4表示审核拒绝
            elif label == 1 or label == 5:
                info['label'] = 'waiting'  # 标签为1或5表示待审
            else:
                info['label'] = 'unknown'  # 其它标签未知
            info['keywords'] = keywords.split()  # 将关键词字符串按照空格分隔为多个关键词列表
            info['row_id'] = i
            info['filename'] = filename
            result.append(info)

    return result


# 精细化调整分词的结果，对标点符号进行更多的处理
def segmentRefine_PuctuationMark(wordsList):
    refineList = []
    for word in wordsList:
        split_words = re.split(
            r"([\s、→↗‧–­…，,。+-­\-\.\"—×\\@%\*:：\?\(\)（）!'【】/><“”=_\^\{\}£،’↓《》■□▲△◆◎●★☆♪✦＝﹝\[\]；;\|­­])",
            word)  # 在标点符号处进行切分
        refineList += split_words
    return refineList


# 精细化调整分词的结果，对数字进行更多的处理
def segmentRefine_Digit(wordsList):
    refineList = []
    for word in wordsList:
        split_words = re.split(r"([\s0-9])", word)
        refineList += split_words
    return refineList


# 精细化调整分词的结果，对中文数字进行更多的处理
def segmentRefine_ChineseDigit(wordsList):
    re_pattern_y = r"\b[零一二三四五六七八九十]{1,4}年\b"
    re_pattern_m = r"\b[零一二三四五六七八九十]{1,2}月\b"
    re_pattern_d = r"\b[零一二三四五六七八九十]{1,3}日\b"
    re_pattern_h = r"\b[零一二三四五六七八九十]{1,3}[时点][钟半]*\b"
    re_pattern_m = r"\b[零一二三四五六七八九十]{1,3}分\b"
    re_pattern_s = r"\b[零一二三四五六七八九十]{1,3}秒\b"
    re_pattern_k = r"\b[零一二三四五六七八九十千百万亿]+\b"
    refineList = []
    for word in wordsList:
        flag_y = re.match(re_pattern_y, word)
        flag_m = re.match(re_pattern_m, word)
        flag_d = re.match(re_pattern_d, word)
        flag_h = re.match(re_pattern_h, word)
        flag_m = re.match(re_pattern_m, word)
        flag_s = re.match(re_pattern_s, word)
        flag_k = re.match(re_pattern_k, word)
        if flag_y or flag_m or flag_d or flag_h or flag_m or flag_s or flag_k:
            split_words = re.split(r"([\s零一二三四五六七八九十千百万亿])", word)
            refineList += split_words
        else:
            refineList.append(word)
    return refineList


# 精细化调整分词的结果，对英文进行更多的处理
def segmentRefine_English(wordsList):
    refineList = []
    for word in wordsList:
        split_words = re.split(r"([\sa-zA-Z])", word)
        refineList += split_words
    return refineList


# 缺省的标记
PAD_token = 0  # 用于对短句进行补零
SOS_token = 1  # 开始标记
EOS_token = 2  # 结束标记
UNKNOWN_token = 3  # 未知标记（用来标记未知符号）


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token, "UNKNOWN": UNKNOWN_token}  # 从词到索引的映射
        self.word2count = {"PAD": 10000, "SOS": 10000, "EOS": 10000, "UNKNOWN": 10000}  # 词的计数
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNKNOWN_token: "UNKNOWN"}  # 从索引到词的映射
        self.num_words = 4  # 初始词汇量为4个词：PAD, SOS, EOS, UNKNOWN

    def addArticle(self, article):  # 将文章中的词加到词汇表中
        for word in article:
            self.addWord(word)

    def addWord(self, word):  # 将词和字加到词汇表中
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

        if len(word) > 1:
            for n in range(len(word)):
                self.addWord(word[n])

    # 去除词频低于min_count的词
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # 重建词汇表
        self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token, "UNKNOWN": UNKNOWN_token}  # 从词到索引的映射
        self.word2count = {"PAD": 10000, "SOS": 10000, "EOS": 10000, "UNKNOWN": 10000}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNKNOWN_token: "UNKNOWN"}  # 从索引到词的映射
        self.num_words = 4  # 初始词汇量为4个词：PAD, SOS, EOS, UNKNOWN

        for word in keep_words:
            self.addWord(word)


# 将语料库中从未在词表中出现的多字词切分为单字词
def splitUnknownWord(corpus_data, voc):
    for x in corpus_data:
        wordList = []
        for word in x['body']:
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
        x['body'] = [x for x in wordList if x != ""]

    return corpus_data

# 将语料库中出现次数低于min_count的多字词拆分为单字词
def splitUncommonWord(corpus_data, voc, min_count):
    for x in corpus_data:
        wordList = []
        for word in x['body']:
            if word in voc.word2count:
                if voc.word2count[word] < min_count and len(word)>1:
                    L = re.split(r"([\W\w])", word)
                    L = [x for x in L if x != ""]
                    wordList += L
                else:
                    wordList.append(word)
            else:
                charList = re.split(r"([\W\w])", word) # 如果不在词表中，直接按字拆分
                charList = [x for x in charList if x != ""]
                for c in charList:
                    if c in voc.word2count:
                        wordList.append(c)
                    else:
                        wordList.append("UNKNOWN")

        x['body'] = [x for x in wordList if x != ""]

    return corpus_data


# 根据语料库建立词汇表
def establishVocabulary(corpus_data, name):
    voc = Vocabulary(name)
    for data in corpus_data:
        voc.addArticle(data['body'])
    return voc


# 根据词汇表，将语料库中出现的未知词替换为UNKONWN
def replaceUnknownWord(corpus_data, voc):
    for data in corpus_data:
        article = data['body']
        article_updated = []
        for word in article:
            if word in voc.word2index:
                article_updated.append(word)
            else:
                article_updated.append("UNKNOWN")
        data['body'] = article_updated
    return corpus_data


# 将语料库转换为数字形式
def transformCorpusToDigit(corpus, voc):
    digit_corpus = []
    for data in corpus:
        article = []
        for word in data['body']:
            article.append(voc.word2index[word])
        article.append(voc.word2index["EOS"])
        if data['label'] == 'pass':
            label = 1
        elif data['label'] == 'reject':
            label = 0
        else:
            continue
        data["article"] = article
        data["label"] = label
        data["body"] = data["body"][0:20]
        digit_corpus.append(data)
    return digit_corpus


# 序列补零的同时时间序列变换为列方向（即行数小的词在前，行数大的词在后，每一列是一个句子）
# L里面是若干个长度不同的序列（每一行表示一个句子，行方向为时间方向，即列代表不同的时间），将这些序列按顺序打包，
# 长度不足的序列补零，直到长度与最长序列相同
def zeroPadding(L, fillvalue=PAD_token):
    # zip_longest函数返回的是一个迭代对象，使用list函数再将它转换为一个list
    return list(itertools.zip_longest(*L, fillvalue=fillvalue))


def printColorAriticle(article):
    color = ['\033[37m','\033[30m','\033[36m','\033[33m','\033[32m','\033[34m','\033[35m','\033[31m']
             # grey      black       qing       yellow     green     blue        manite     red
    ave_value = 1 / len(article)
    for x in article:
        if x[1] <= 1 * ave_value:
            print(color[0] + x[0],end='')
        elif x[1] <= 2 * ave_value:
            print(color[1] + x[0], end='')
        elif x[1] <= 3 * ave_value:
            print(color[2] + x[0], end='')
        elif x[1] <= 4 * ave_value:
            print(color[3] + x[0], end='')
        elif x[1] <= 5 * ave_value:
            print(color[4] + x[0], end='')
        elif x[1] <= 6 * ave_value:
            print(color[5] + x[0], end='')
        elif x[1] <= 7 * ave_value:
            print(color[6] + x[0], end='')
        else:
            print(color[7] + x[0], end='')
    print()


