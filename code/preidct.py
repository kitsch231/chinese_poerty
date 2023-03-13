import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import sys
import torch
import numpy as np
import re
from models import Mybert
from tensorboardX import SummaryWriter
from untils import My_Dataset,get_time_dif,generate_random,generate_word
from models import *
from config import Config
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer



if __name__=='__main__':
    config=Config()
    mynet = Mybert(config)
    ## 模型放入到GPU中去

    all_pre0=[]
    all_pre1=[]
    mynet = mynet.to(config.device)
    print(config.save_path)
    mynet.load_state_dict(torch.load(config.save_path))

    po='一二三四'
    pohead='失败'
    res=generate_random(mynet,config,po)
    res_head=generate_word(mynet,config,pohead)
    print('提示词：{},生成诗词：{}'.format(po,res))
    print('藏头诗提示词：{},生成诗词：{}'.format(pohead,res_head))
    r = re.split(r"([.。!！?？；;，,\s+])", res)[:-1]
    print(r)
    for x in range(0,len(r),2):
        print(r[x]+r[x+1])
