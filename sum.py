# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : sum.py
# Time       ：2023/4/18 22:05
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import pandas as pd

# 读取数据
dfcf = pd.read_csv('Output/DFCF_Clean.csv')
trad = pd.read_csv('input/000001SH_Trad.csv')

'''
# 遍历trad的日期列，计算每个日期对应的Influence总和
influence_sums = []
for date in trad['DateTime']:
    sum_of_date = dfcf.loc[dfcf['PostingTime'].astype(str).str.startswith(str(date)), 'Influence'].sum()
    influence_sums.append(sum_of_date)

# 将Influence总和添加到trad中
trad['InfluSum'] = influence_sums

# 将结果保存到文件中
trad.to_csv('input/000001SH_Trad.csv', index=False)
'''

# Clean the date in comment csv
# 将日期列转换为字符串格式
dfcf['PostingTime'] = dfcf['PostingTime'].astype(str)
trad['DateTime'] = trad['DateTime'].astype(str)

# 只取两个DataFrame共同的时间
common_time = list(set(trad['DateTime']).intersection(set(dfcf['PostingTime'].str[:10])))

# 取出共同时间对应的数据并保存为新文件
dfcf_date_clean = dfcf[dfcf['PostingTime'].str[:10].isin(common_time)]

dfcf_date_clean.to_csv('Output/DFCF_DateClean.csv', index=False,encoding = 'utf-8-sig')
