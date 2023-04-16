# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : WorldSplit.py
# Time       ：2023/4/15 16:45
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import os

import pandas as pd
import jieba

# Define the paths to custom stopwords and dictionary files
stopword_path = 'Senti_Dic/stopwords.txt'

# Load custom stopwords files into a set
stop_words = set()
with open(stopword_path, 'r', encoding='utf-8') as f:
    words = f.read().split()
    stop_words.update(words)

# Add custom stopwords and dictionary to jieba
for word in stop_words:
    jieba.add_word(word)

# load the new dictionary into jieba
jieba.load_userdict(r"Senti_Dic/Stock_Words.txt")

# Load the CSV file
data = pd.read_csv('Data/DFCF_Date_Clean_2021.csv')

# Create a new dataframe with a 'WorldSplitResult' column
new_data = pd.DataFrame(columns=['WordSplitResult'])

# Preprocess the comments
for i in range(len(data)):
    # Remove noise in the comments
    comment = str(data['Title'][i])  # convert to string type

    # Split the comment into words
    words = jieba.lcut(comment, cut_all=False)  # use exact mode
    # Remove stop words, punctuation, and other unnecessary characters
    words = [w for w in words if w not in stop_words and w != ' ']
    # remove numbers
    words = [w for w in words if not any(char.isdigit() for char in w)]

    # Save the results of word segmentation into the new dataframe
    new_data.loc[i] = [' '.join(words)]

# Merge the original dataframe with the new dataframe containing the segmented comments
data = pd.concat([data, new_data], axis=1)

# Save the modified dataframe to a new CSV file
data.to_csv('Output/CommentSplit_2021.csv', sep=',', index=False, encoding='utf-8-sig')