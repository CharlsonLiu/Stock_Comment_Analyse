# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : word2vec.py
# Time       ：2023/4/18 22:40
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import gensim
import numpy as np
import torch

# 加载word2vec模型
model_path = 'params/word2vec/model.bin'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# 读取DFCF_DateClean.csv的SplitWords
with open('Output/DFCF_DateClean.csv', 'r', encoding='utf-8') as f:
    data = f.readlines()

# 逐行处理并保存为numpy数组
word_vecs = []
for line in data:
    words = line.strip().split()
    vecs = []
    for word in words:
        try:
            vec = word2vec[word]
        except KeyError:
            vec = np.zeros(100)
        vecs.append(vec)
    vecs = np.array(vecs)
    # 加入开始标志和结束标志
    vecs = np.vstack([np.zeros((1, 100)), vecs, np.zeros((1, 100))])
    word_vecs.append(vecs)
word_vecs = np.array(word_vecs)

# 转换为torch.Tensor并保存为文件
word_vecs = torch.from_numpy(word_vecs).float()
torch.save(word_vecs, 'input/word_vecs.pt')
