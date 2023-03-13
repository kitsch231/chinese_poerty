import os
import random

import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import cv2
from config import Config
import json
# a.通过词典导入分词器
#"bert-base-chinese"

#bert_model/chinese-bert-wwm-ext

#tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
class My_Dataset(Dataset):
    def __init__(self,path,config,iftrain):#### 读取数据集
        self.config=config
        #启用训练模式，加载数据和标签
        self.iftrain=iftrain
        self.df = pd.read_csv(path)
        self.text = self.df['text'].to_list()

        with open('data/word_idx.josn', 'r',encoding='utf-8') as f:
            word_idx=json.load(f)
        with open('data/idx_word.josn', 'r',encoding='utf-8') as f:
            idx_word=json.load(f)
        self.word_idx=word_idx
        self.idx_word=idx_word

    def __getitem__(self, idx):
        sen=self.text[idx]
        text='s'+sen[:-1]+'e'
        label='s'+sen[1:]+'e'

        if len(text)>=self.config.pad_size:
            text=text[:self.config.pad_size]
            #print(len(sen))
        else:
            text=text+'o'*(self.config.pad_size-len(text))

        if len(label)>=self.config.pad_size:
            label=label[:self.config.pad_size]
            #print(len(sen))
        else:
            label=label+'o'*(self.config.pad_size-len(label))
        # print(text)
        # print(label)
        text=[self.word_idx[x] for x in text]
        label=[self.word_idx[x] for x in label]
        #print(text)
        # 中文-英文  （t1[我 吃 饭],t2[i eat food]）  [[0,0,0,0,0],[1,1,1,1,1]]
        #text 三个部分  token_type_ids(句子对 中文句子 英文句子)
        input_id= torch.tensor(text, dtype=torch.long)
        label=torch.tensor(label, dtype=torch.long)
        #attention_mask = torch.tensor(text['attention_mask'], dtype=torch.long)#可用可不用
        #
        return input_id.to(self.config.device),label.to(self.config.device)

    def __len__(self):
        return len(self.df)#总数据长度




def get_time_dif(start_time):

    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def en_text(text,config):
    with open('data/word_idx.josn', 'r', encoding='utf-8') as f:
        word_idx = json.load(f)

    text='s'+text+'e'
    if len(text) >= config.pad_size:
        text = text[:config.pad_size]
        # print(len(sen))
    else:
        text = text + 'o' * (config.pad_size - len(text))

    tlist =[]

    for x in text:
        try:
            tlist.append(word_idx[x])
        except:
            print(random.choice(word_idx.values()))
            tlist.append(random.choice(word_idx.values()))
    input= torch.tensor(tlist, dtype=torch.long)
    return input.to(config.device)

def de_text(text,config):
    text=text.tolist()
    with open('data/idx_word.josn', 'r', encoding='utf-8') as f:
        idx_word = json.load(f)

    text = [idx_word[str(x)] for x in text]
    return ''.join(text)

def generate_random(model,config,start='可怜'):
    """自由生成一首诗歌"""
    model.eval()
    poetry = []
    sentence_len = 0

    hidden = model.init_hidden(2, 1)
    with torch.no_grad():
        for i in range(config.pad_size):
            input = en_text(start,config)
            input = input.unsqueeze(0)
            #print(input)
            output,hidden=model(input,hidden)

            hidden = ([h.data for h in hidden])
            top_index =torch.max(output.data,1)[1].cpu().numpy()

            #print(top_index)
            w=de_text(top_index,config)

            w=w.split('e')[0][-1]
            if w=='o':
                with open('data/word_idx.josn', 'r', encoding='utf-8') as f:
                    word_idx = json.load(f)

                w=random.choice(list(word_idx.keys())[3:])
            start=start+w
            # print(torch.max(output.data,1)[1].cpu())
            #print(start)
            if w=='e':
                break

            if w in ['。', '!',',','?']:
                sentence_len += 1
                if sentence_len == config.max_sen:
                    poetry.append(w)
                    break
            poetry.append(w)

        # print('*************************')

    return start

def generate_word(model,config,start_list='可怜'):
    """自由生成一句诗歌"""
    model.eval()
    poetry = []
    sentence_len = 0

    start=start_list[0]
    with torch.no_grad():
        for i in range(config.pad_size):

            hidden = model.init_hidden(2, 1)
            input = en_text(start,config)
            input = input.unsqueeze(0)
            #print(input)
            output,hidden=model(input,hidden)

            hidden = ([h.data for h in hidden])
            top_index =torch.max(output.data,1)[1].cpu().numpy()

            #print(top_index)
            w=de_text(top_index,config)
            w=w.split('e')[0][-1]
            start=start+w
            # print(torch.max(output.data,1)[1].cpu())
            #print(start)
            if w=='e':
                break

            if w in ['。', '!',',','?']:
                sentence_len += 1
                if len(start_list)>sentence_len>0:
                    start=start+start_list[sentence_len]
                if sentence_len == len(start_list):
                    poetry.append(w)
                    break

            poetry.append(w)

            # print('*************************')

    return start


if __name__=='__main__':
    config=Config()
    train_data=My_Dataset('data/train.csv',config,1)
    train_iter = DataLoader(train_data, batch_size=1)
    n=0
    for x,y in train_iter:

        print(x.shape)
        print(y.shape)
        # #print(y)
        # print('************')
        break
