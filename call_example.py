# -*- coding: UTF-8 -*-

import requests

# 利用POST接口调用地址"http://localhost:5000/predict"提供的服务，被审核的文件保存为文本文件，在调用时给出其路径即可，代码如下
resp = requests.post("http://10.96.200.32:6000/predict", params={"article": """回复内容：是不是错过了有效发布时间啊？<br/><br/><br/>回复附件："""})

# 返回内容为json串，第1个字段predict_label，predict_label=1表示审核通过，predict_label=0表示审核拒绝。
# 第2个字段给出了每个词对应的分值，例如[['这是',1],['我',2],['的',1],['玩具',8]]

print(resp.json(), "\n\n\n") # this is a dictionary, show as single-quots
print(resp.text, "\n\n\n") # get double-quots
