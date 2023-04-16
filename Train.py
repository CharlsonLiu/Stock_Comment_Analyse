# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Train.py
# Time       ：2023/4/16 17:50
# Author     ：Liu Ziyue
# version    ：python 3.8
# Description：
"""
import argparse
import os
import random

import gensim
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score,precision_score
from tqdm import tqdm

from models import LSTM, LSTMWithAttention, CNN, BiLSTM, CNNwithLSTM
from LoadData import LoadData

# load word2vec model
word2vec_model_path = "params/word2vec/word2vec.model"
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)

# 设置超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 10
vocab_size = 10000  # size of vocabulary
embedding_dim = 100  # size of word embeddings
output_dim = 1  # size of model output
hidden_dim = 128  # size of hidden layer
num_layers = 2  # number of layers in model
num_filters = 100  # number of filters in CNN model
filter_sizes = [3, 4, 5]  # filter sizes in CNN model
weights = torch.FloatTensor(word2vec_model.wv.vectors)  # pre-trained word embeddings
dropout = 0.5  # dropout rate in CNNwithLSTM model

# training hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10
patience = 3  # early stopping patience

# loss function
criterion = nn.BCEWithLogitsLoss()

# 设置随机种子，保证实验可重复性
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 选用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 导入数据
data_loader = LoadData(embedding_file_path='params/word2vec/word2vec.model',
                        stop_words_file_path='Senti_Dic/stopwords.txt',
                        user_dict_file_path='Senti_Dic/Stock_Words.txt')
data_file_path = 'data/sentiment_data.csv'
vectors, labels = data_loader.load_data(data_file_path)

# 随机分配数据集
train_vectors, val_vectors, train_labels, val_labels = train_test_split(vectors, labels, test_size=0.2, random_state=42)
train_dataset = torch.utils.data.TensorDataset(train_vectors, train_labels)
val_dataset = torch.utils.data.TensorDataset(val_vectors, val_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_items = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_items += inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    avg_loss = total_loss / total_items
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return avg_loss, accuracy, recall, f1

def save_metrics_to_csv(model_name, mode, epoch, loss, accuracy):
    # Add the following lines of code
    validation_labels = []
    predicted_labels = []
    model_name.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model_name(inputs)
            _, predicted = torch.max(outputs.data, 1)
            validation_labels += labels.cpu().numpy().tolist()
            predicted_labels += predicted.cpu().numpy().tolist()

    recall_val = recall_score(validation_labels, predicted_labels, average='macro')
    f1_val = f1_score(validation_labels, predicted_labels, average='macro')
    # Compute precision, recall, and f1-score
    y_true = np.array(validation_labels)
    y_pred = np.array(predicted_labels)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # Construct metrics dataframe
    metrics_df = pd.DataFrame({'epoch': epoch,
                               'loss': loss,
                               'accuracy': accuracy,
                               'precision': precision,
                               'recall': recall,
                               'f1': f1}, index=[0])

    # Define csv file path and write to csv
    csv_file_path = f'params/{model_name}/{mode}/metrics.csv'
    if not os.path.exists(os.path.dirname(csv_file_path)):
        os.makedirs(os.path.dirname(csv_file_path))
    if not os.path.exists(csv_file_path):
        metrics_df.to_csv(csv_file_path, index=False)
    else:
        metrics_df.to_csv(csv_file_path, mode='a', header=False, index=False)

def train(model, optimizer, criterion, train_loader, valid_loader, device, model_path):
    # define lists to store the loss and accuracy
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    # define the best accuracy and corresponding epoch
    best_acc = 0.0
    best_epoch = -1

    for epoch in range(1, num_epochs+1):
        # define variables to record the training loss and correct predictions
        train_loss , train_total , train_correct = 0.0 , 0.0 , 0.0
        # switch to train mode
        model.train()

        # initialize the progress bar
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training", unit="batch")

        # iterate over batches of training data
        for batch in train_progress_bar:
            # move the batch to the device
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            # record the loss and number of correct predictions
            train_loss += loss.item() * len(inputs)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # update the progress bar
            train_progress_bar.set_postfix({
                "loss": loss.item(),
                "accuracy": train_correct / train_total,
            })

        # calculate the average training loss and accuracy for this epoch
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Evaluate on validation set
        valid_loss, valid_acc, valid_recall, valid_f1 = evaluate(model, valid_loader, criterion, device)

        # print the training and validation metrics
        print(f"Epoch {epoch} training loss: {train_loss:.6f} accuracy: {train_acc:.6f}")
        print(f"Epoch {epoch} validation loss: {valid_loss:.6f} accuracy: {valid_acc:.6f}")

        # save the model weights if the validation accuracy improves
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        # save the training and validation metrics to a CSV file
        save_metrics_to_csv(model, 'train', epoch+1, train_loss/len(train_loader), None)
        save_metrics_to_csv(model, 'val', epoch+1, valid_loss, valid_acc, valid_recall, valid_f1)

    # print the best validation accuracy and corresponding epoch
    print(
        f'Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.3f}, Valid Loss: {valid_loss:.3f},'
        f' Valid Acc: {valid_acc:.3f}, Valid Recall: {valid_recall:.3f}, Valid F1: {valid_f1:.3f}')

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train model on text classification task')
    parser.add_argument('--model', type=int, default=1, help='Choose model to use (1-5)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='gpu', help='Device to train on (cpu or cuda)')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to data folder')
    parser.add_argument('--model_path', type=str, default='params/', help='Path to save model parameters')
    parser.add_argument('--csv_path', type=str, default='params/', help='Path to save CSV files')

    args = parser.parse_args()

    # Define hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Convert data into PyTorch tensors and create DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_vectors, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_vectors, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer
    model_choice = input()
    model = choose_model()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model and save best weights
    train(model, optimizer, num_epochs,  train_loader, val_loader, device)

def choose_model(model_choice):
    if model_choice == 1:
        return LSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
             num_layers=num_layers, output_dim=output_dim)
    elif model_choice == 2:
        return LSTMWithAttention(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                          num_layers=num_layers, output_dim=output_dim)
    elif model_choice == 3:
        return CNN(vocab_size=vocab_size, embedding_dim=embedding_dim, num_filters=num_filters,
            filter_sizes=filter_sizes, output_dim=output_dim)
    elif model_choice == 4:
        return BiLSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
               num_layers=num_layers, output_dim=output_dim)
    elif model_choice == 5:
        return CNNwithLSTM(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size,
                    num_classes=output_dim, weights=weights, kernel_sizes=[3, 4, 5], dropout=0.5)
    else:
        print("Invalid choice of model, please choose between 1-5.")
        exit()


if __name__ == '__main__':
    main()

