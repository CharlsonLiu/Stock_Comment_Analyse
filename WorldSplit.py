# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : WorldSplit.py
# Time       ：2023/4/15 16:45
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import re
import pandas as pd
import jieba


def Send2Jieba(stopwords_path, userdict_path):
    # Load custom stopwords files into a set
    stop_words = set()
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        words = f.read().split()
        stop_words.update(words)

    # Add custom stopwords and dictionary to jieba
    for word in stop_words:
        jieba.add_word(word)

    # load the new dictionary into jieba
    jieba.load_userdict(userdict_path)

    # return the customized jieba function
    return jieba,stop_words

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
    return text


def CsvSplit(csv_path, jieba_func, stop_words, txt_path):
    # Load the CSV file
    data = pd.read_csv(csv_path)

    # Create a new dataframe with a 'SplitWords' column
    new_data = pd.DataFrame(columns=['SplitWords'])

    # Preprocess the comments and save the segmented words to a txt file
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i in range(len(data)):
            # Remove noise in the comments
            comment = str(data['Title'][i])  # convert to string type

            comment = clean_text(comment)

            # Split the comment into words using the customized jieba function
            words = jieba_func.lcut(comment, cut_all=False)  # use exact mode

            # Remove stop words, punctuation, and other unnecessary characters
            words = [w for w in words if w.strip() and w not in stop_words and not any(char.isdigit() for char in w)]
            # Save the results of word segmentation into the new dataframe
            if words:
                new_data.loc[i] = [' '.join(words)]
                f.write(' '.join(words) + '\n')
                # data.loc[i, 'Weight'] = weight
            else:
                # Remove the row if there is no word after segmentation
                data.drop(i, inplace=True)

    # Merge the original dataframe with the new dataframe containing the segmented comments
    data = pd.concat([data, new_data], axis=1)

    # Save the modified dataframe to a new CSV file
    #data.to_csv('Output/DFCF_Clean.csv', sep=',', index=False, encoding='utf-8-sig')

def TextSplit(txt_path, jieba_func , stop_words):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split the text into words using the customized jieba function
    words = jieba_func.lcut(text, cut_all=False)  # use exact mode

    # Remove stop words, punctuation, and other unnecessary characters
    words = [w for w in words if w.strip() and w not in stop_words and not any(char.isdigit() for char in w)]

    # Save the results of word segmentation into a new text file
    with open('Output/Text_Split.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(words))


def main(stopwords_path, userdict_path, csv_path , text_path):
    jieba_func,stopwords = Send2Jieba(stopwords_path, userdict_path)
    CsvSplit(csv_path, jieba_func , stopwords , txt_path = text_path)
    #TextSplit(txt_path, jieba_func , stopwords)


if __name__ == '__main__':
    stopwords_path = 'Senti_Dic/stopwords.txt'
    userdict_path = 'Senti_Dic/Stock_Words.txt'
    csv_path = 'Data/工作簿1.csv'
    txt_path = 'Output/DFCF_words1.txt'
    main(stopwords_path,userdict_path,csv_path , text_path=txt_path)
