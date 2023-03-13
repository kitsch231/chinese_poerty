import torch.nn as nn
from transformers import BertModel,BertForNextSentencePrediction
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


class Mybert(nn.Module):
    def __init__(self,config):
        super(Mybert, self).__init__()
        self.config=config
        self.hidden_dim =512
        self.emb=nn.Embedding(config.n_vocab,128)
        self.lstm = nn.LSTM(128,self.hidden_dim, num_layers=2,batch_first=True)
        self.fc = nn.Linear(self.hidden_dim,self.config.n_vocab)
        self.drop=nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,inx,h):

        outputs = self.emb(inx)
        x, hidden =self.lstm(outputs,h)
        x=x.reshape(self.config.pad_size*x.shape[0],-1)
        x = self.fc(x)
        return x,hidden

    def init_hidden(self, layer_num, batch_size):
        return (Variable(torch.zeros(layer_num, batch_size, self.hidden_dim)).to(self.config.device),
                Variable(torch.zeros(layer_num, batch_size, self.hidden_dim)).to(self.config.device))

