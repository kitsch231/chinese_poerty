import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,BertModel
from config import Config
from tqdm import tqdm
import json

config=Config()

#重新整理一下数据集
with open('data/poetry.txt','r',encoding='utf-8')as f:
    texts=f.readlines()
words=[]

df=[]
for x in tqdm(texts):
    if '_' not in x:
        x=x.replace('，',',').replace('：', ':').replace('？','?').replace('！','!').strip()
        if '（' in x or '_' in x:
            x = re.sub(r'\（.*\）', '', x)
        x=x.split(':')
        if len(x)!=2:
            pass
        elif 10<len(x[1])<=128:
            r1 = r'[a-zA-Z0-9]'
            x[1]= re.sub( r1, '', x[1])
            x_list=re.split("[,|.|?|!|。]", x[1])[:-1]
            x_list=[len(x) for x in x_list]
            result = x_list.count(x_list[0]) == len(x_list)
            for w in x[1]:
                if w not in words:
                    words.append(w)
            if result:
                df.append(x)


words=set(words)
wordidx={}#词,数字索引
wordidx['s']=0
wordidx['e']=1
wordidx['o']=2

n=3
for x in words:
   wordidx[x]=n
   n=n+1
print(wordidx)
idxword={}##数字，词索引
for x,y in wordidx.items():
    idxword[y]=x
print(idxword)

with open('data/word_idx.josn','w',encoding='utf-8')as f:
    f.write(json.dumps(wordidx,ensure_ascii=False))
with open('data/idx_word.josn','w',encoding='utf-8')as f:
    f.write(json.dumps(idxword,ensure_ascii=False))

df=pd.DataFrame(df)
df.columns=['title','text']
print(df)
train,_=train_test_split(df,test_size=0.1)
val,test=train_test_split(_,test_size=0.5)
print(val)
train.to_csv('data/train.csv',index=None)
test.to_csv('data/test.csv',index=None)
val.to_csv('data/val.csv',index=None)