# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Word2Vec_Train.py
# Time       ：2023/4/18 16:02
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
from random import randint

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from gensim.models import Word2Vec
from matplotlib import cm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm


warnings.filterwarnings('ignore')

# 读入已经分好词的文本
with open('input/Words_4_Train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 去除每行开头和结尾的空格，并合并成一个字符串
corpus = ' '.join([line.strip() for line in lines])

# 统计每个词的出现频率
word_freq_dict = {}
for word in corpus.split():
    if word in word_freq_dict:
        word_freq_dict[word] += 1
    else:
        word_freq_dict[word] = 1

# 将文本转化为句子列表
sentences = [jieba.lcut(line.strip()) for line in lines]

# 定义一个函数来生成渐变颜色
def generate_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # 生成三个随机数
    r, g, b = [randint(200, 255) for _ in range(3)]
    return f"rgb({r},{g},{b})"


# 定义词向量维度和窗口大小
embedding_size = 100
window_size = 10
seed = 123
# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size = embedding_size, window=window_size, min_count=1 , workers=64 , seed = seed)

# 保存模型
model.save('params/word2vec/model.bin')

plt.figure(1)
# 定义词云图参数
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei
# 选出出现频率最高的前40个词
top_words = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
top_words_dict = dict(top_words)


# 生成词云图
wc = WordCloud(
    width=600, height=400,
    background_color='Black',
    max_font_size=100,
    font_path = 'C:\Windows\Fonts\msyhl.ttc',
    color_func = generate_color_func,
    random_state=50
).generate_from_frequencies(top_words_dict)

# 将词云图转为numpy array，将背景设置为透明
wc_array = np.array(wc)
if wc_array.ndim == 3 and wc_array.shape[2] == 4:
    wc_array[wc_array[:, :, 3] == 0] = [255, 255, 255, 0]
wc_array = wc_array.astype(np.uint8)

# 绘制词云图
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(wc_array, interpolation='bilinear')
ax.axis('off')
plt.title("股评文本词云图",fontsize = 25)
plt.savefig("graphs/wordcloud.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()



# 获取所有词向量
word_vectors = model.wv
# 取出词向量中前200个最有代表性的词
words = [word for word, _ in word_vectors.most_similar()[:200]]
vectors = [word_vectors[word] for word in words]

# 降维并可视化词向量
word_vectors_tsne_2d = TSNE(n_components=2, random_state=seed).fit_transform(vectors)

# 绘制词向量分布图
plt.figure(2)

# 设置colormap
cmap = cm.get_cmap('coolwarm')

# 绘制2D散点图
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(word_vectors_tsne_2d[:, 0], word_vectors_tsne_2d[:, 1], c=range(len(words)), cmap='hsv')
cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(words)))
cbar.ax.set_yticklabels(words)
# 在每个点上添加标签和阴影
for i, word in enumerate(words):
    ax.annotate(word, (word_vectors_tsne_2d[i, 0], word_vectors_tsne_2d[i, 1]),
                fontsize=10, fontfamily='Arial', fontweight='bold',
                color='white', shadow=False)


plt.title('词向量2D点图', fontsize=20)
plt.axis('off')
plt.savefig("graphs/wordvector2D.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

plt.figure(3)
# 将词向量降至三维
word_vectors_tsne_3d = TSNE(n_components=3, random_state=seed).fit_transform(vectors)

# 绘制3D散点图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D散点图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(word_vectors_tsne_3d[:, 0], word_vectors_tsne_3d[:, 1], word_vectors_tsne_3d[:, 2], c=range(len(words)), cmap='hsv')
cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(words)))
cbar.ax.set_yticklabels(words)
for i, word in enumerate(words):
    ax.text(word_vectors_tsne_3d[i, 0], word_vectors_tsne_3d[i, 1], word_vectors_tsne_3d[i, 2], word,
            fontsize=10, fontfamily='Arial', fontweight='bold',
            color='white', backgroundcolor='black')


plt.title('词向量3D散点图', fontsize=20)
plt.axis('off')
plt.savefig("graphs/wordvector3D.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
