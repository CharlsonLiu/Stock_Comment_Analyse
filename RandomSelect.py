# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : RandomSelect.py
# Time       ：2023/4/18 21:49
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import pandas as pd
import numpy as np

# 读取CSV文件
data = pd.read_csv('Output/DFCF_Clean.csv')

# 随机选取3000行作为训练集
train_data = data.sample(n=3000, random_state=420, replace=False)
train_data.to_csv('input/train.csv', index=False,encoding='utf-8-sig')

# 随机选取1500行作为测试集
test_data = data.loc[~data.index.isin(train_data.index)]
test_data = test_data.sample(n=1500, random_state=420, replace=False)
test_data.to_csv('input/test.csv', index=False,encoding='utf-8-sig')

# 随机选取1500行作为验证集
val_data = data.loc[~data.index.isin(train_data.index)]
val_data = val_data.loc[~val_data.index.isin(test_data.index)]
val_data = val_data.sample(n=1500, random_state=420, replace=False)
val_data.to_csv('input/val.csv', index=False,encoding='utf-8-sig')
