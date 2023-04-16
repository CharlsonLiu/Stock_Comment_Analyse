# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : models.py
# Time       ：2023/4/16 16:41
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import torch
import torch.nn as nn
from torch.autograd.grad_mode import F


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output


class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)

        # 计算attention
        attn_weights = self.attention(output).squeeze(2)
        soft_attn_weights = nn.functional.softmax(attn_weights, dim=1)
        context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        # 拼接context和hidden并进行线性变换
        output = torch.cat((context, hidden[-1]), dim=1)
        output = self.fc(output)
        return output


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        output = self.fc(cat)
        return output


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.fc(hidden)
        return output


class CNNwithLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes, weights, kernel_sizes=[3, 4, 5],
                 dropout=0.5):
        super(CNNwithLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=ks)
            for ks in kernel_sizes
        ])
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # permute to [batch_size, embedding_dim, seq_len]
        # apply convolutional layer with different kernel sizes
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved shape: [(batch_size, hidden_dim, conv_seq_len), ...] for each kernel size
        # apply max pooling over time dimension
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled shape: [(batch_size, hidden_dim), ...] for each kernel size
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat shape: [batch_size, hidden_dim * len(kernel_sizes)]
        lstm_out, _ = self.lstm(cat.unsqueeze(1))
        # lstm_out shape: [batch_size, 1, hidden_dim * 2]
        lstm_out = lstm_out.squeeze(1)
        # lstm_out shape: [batch_size, hidden_dim * 2]
        output = self.fc(lstm_out)
        # out shape: [batch_size, num_classes]
        return output
