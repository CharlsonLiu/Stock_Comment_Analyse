# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Clean.py
# Time       ：2023/4/14 11:02
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import pandas as pd
import re
import jieba

# define a function to clean up text columns
def clean_text(text):
    # remove URLs
    text = re.sub(r'http\S+', '', text)

    # remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # remove mentions and hashtags
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # load the new dictionary into jieba
    jieba.load_userdict(r"Senti_Dic/Stock_Words.txt")

    # remove stopwords
    stopwords = set()
    with open('Senti_Dic/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    # Split the comment into words
    words = jieba.lcut(text, cut_all=False)  # use exact mode

    # Remove stop words, punctuation, and other unnecessary characters
    words = [w for w in words if w not in stopwords and w != ' ']
    # remove numbers
    words = [w for w in words if not any(char.isdigit() for char in w)]

    # rejoin words into a cleaned text string
    text = ' '.join(words)

    return text

def main():
    # read in the original comment data CSV file
    comments_df = pd.read_csv("Data/DFCF_SH.csv", encoding='utf-8')

    # read in the SSE000001 CSV file
    ssec_df = pd.read_csv("input/000001SH_Trad.csv")

    # filter the comments based on SSE000001 trading dates
    ssec_dates = set(ssec_df["DateTime"])
    comments_df = comments_df[comments_df["PostingTime"].isin(ssec_dates)]

    # clean up the "CommentText" column
    comments_df["Title"] = comments_df["Title"].apply(clean_text)

    # save the cleaned data to a new CSV file
    comments_df.to_csv("Output/DFCF_Clean.csv", index=False, encoding='utf-8-sig', sep=',')

if __name__ == "__main__":
    main()