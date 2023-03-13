import streamlit as st
from untils import *
from models import *
from config import Config
import re
#在终端输入streamlit run ui.py打开界面
st.title('古诗生成器')

config = Config()
mynet = Mybert(config)
## 模型放入到GPU中去


#四个选项
op=st.sidebar.selectbox( '请选择生成方式：',('提示词生成全诗','提示词生成藏头诗'))
st.write('当前功能:', op)
if op=='提示词生成全诗':
    mynet = mynet.to(config.device)
    print(config.save_path)
    mynet.load_state_dict(torch.load(config.save_path))
    text = st.text_input(label='请输入文本用于生成')  # 标题
    if st.button('点击开始生成'):#确认按键
        res=generate_random(mynet,config,text)
        r = re.split(r"([.。!！?？；;，,\s+])", res)[:-1]
        for x in range(0, len(r), 2):
            st.text(r[x] + r[x + 1])

if op=='提示词生成藏头诗':
    mynet = mynet.to(config.device)
    print(config.save_path)
    mynet.load_state_dict(torch.load(config.save_path))
    text = st.text_input(label='请输入文本用于生成')  # 标题
    if st.button('点击开始生成'):#确认按键
        res=generate_word(mynet,config,text)
        r = re.split(r"([.。!！?？；;，,\s+])", res)[:-1]
        for x in range(0, len(r), 2):
            st.text(r[x] + r[x + 1])