import requests

resp = requests.post("http://localhost:5000/predict", files={"file": open('/home/zhangyong/temp.txt','r')})

print(resp.json())