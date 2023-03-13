import os.path
import torch
import time
'''换网络运行只需要更换self.mynet=这个参数即可，其他根据情况微调'''

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.3     # 随机失活
        self.require_improvement = 20000  # 若超过2000batch效果还没提升，则提前结束训练
        #self.num_classes1= 3# 第二个标签类别数，无需修改
        self.num_epochs = 50   # epoch数
        self.pad_size =128# 每句话处理成的长度(短填长切) 根据平均长度来
        self.learning_rate =1e-3#学习率
        self.frac=1#使用数据的比例，因为训练时间长，方便调参使用,1为全部数据，0.1代表十分之一的数据
        self.embed =128
        self.n_vocab=6455#词表大小
        self.mynet='lstm'
        self.max_sen=16#最大生产四句话的诗，太长容易出问题
        self.batch_size =32 # mini-batch大小，看显存决定
        if not os.path.exists('model'):
            os.makedirs('model')

        self.save_path = 'model/'+self.mynet+'.pt'##保存模型的路径
        self.log_dir= './log/'+self.mynet+'/'+str(time.time())#tensorboard日志的路径


