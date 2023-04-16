# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : embedding.py
# Time       ：2023/4/16 16:32
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""

from gensim.models import Word2Vec
import jieba
import pandas as pd

# 读取数据
df = pd.read_csv('train.csv')

# 加载停用词
with open('Senti_Dic/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])

# 加载自定义词典
jieba.load_userdict('Senti_Dic/Stock_Words.txt')

# 对评论进行分词，并去除停用词
texts = [[word for word in jieba.lcut(text,cut_all = False) if word not in stopwords] for text in df['Title']]

# 训练词向量模型
model = Word2Vec(texts, size=100, window=5, min_count=5)

# 保存模型
model.save('params/word2vec/word2vec.model')

