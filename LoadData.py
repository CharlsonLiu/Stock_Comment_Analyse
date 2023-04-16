# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : LoadData.py
# Time       ：2023/4/16 13:45
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import pandas as pd
import jieba
import numpy as np
from gensim.models import Word2Vec
import torch


class LoadData:
    def __init__(self, embedding_file_path, stop_words_file_path, user_dict_file_path):
        self.embedding_model = Word2Vec.load(embedding_file_path)
        self.stop_words_file_path = stop_words_file_path
        self.user_dict_file_path = user_dict_file_path
        self.stop_words = None
        self.user_dict = None
        self.load_stop_words()
        self.load_user_dict()

    def load_stop_words(self):
        if self.stop_words_file_path:
            with open(self.stop_words_file_path, 'r', encoding='utf-8') as f:
                self.stop_words = [line.strip() for line in f.readlines()]

    def load_user_dict(self):
        if self.user_dict_file_path:
            jieba.load_userdict(self.user_dict_file_path)

    def preprocess(self, text):
        text = jieba.lcut(text, cut_all=False)
        if self.stop_words:
            text = [word for word in text if word not in self.stop_words]
        text = [word for word in text if word in self.embedding_model.wv.vocab]
        return text

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        texts = data['title'].apply(self.preprocess).tolist()
        labels = data['senti_score'].tolist()
        max_len = max(len(text) for text in texts)
        vectors = np.zeros((len(texts), max_len, self.embedding_model.vector_size))
        for i, text in enumerate(texts):
            for j, word in enumerate(text):
                vectors[i, j] = self.embedding_model.wv[word]
        vectors = torch.from_numpy(vectors).float()
        labels = torch.tensor(labels)
        return vectors, labels


