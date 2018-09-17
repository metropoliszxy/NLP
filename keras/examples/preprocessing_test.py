#coding=utf-8
#代码中包含中文，就需要在头部指定编码。

'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

from __future__ import print_function
 

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
#import pymysql
import jieba
#from bs4 import BeautifulSoup
import re
from keras.models import model_from_json
import os
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('0 ---> Loading data...')

type = ['游戏活动资讯', '游戏直播资讯', '赛事资讯', '赛事解读', '赛事采访', '赛事回顾', '赛事竞猜、博彩', '版本、服务更新','游戏攻略', '游戏教学', '战队动态', '人物采访', '人物群访', '电竞主播', '电竞选手', '娱乐八卦', '业界新闻, 电子竞技体育','产业报告', '电竞发展趋势', '电竞立法', '人工智能电竞', '电竞峰会', '电竞奥运', '颁奖盛典', '电竞协会']
list13 = []     # 存放序列全为0的原始list [0, 0, 0, ....]
list14 = []     # 存放 每一篇文章类别 的list，[[], [], [], ....]
article_list = []   # 存放每一篇符合要求的文章
tag_list = []   # 存放每一篇文章的标签
word_dict = []  # 词字典



