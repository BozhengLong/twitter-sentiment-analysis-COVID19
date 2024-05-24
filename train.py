"""
Author: Bozheng Long
Created Date: 2024-05-22
Last Modified Date: 2024-05-24
Description:
This file is used to train a simple neural network model to predict the emotion 
labels of the comments in a YouTube video.
"""
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torchtext
from torchtext.data import get_tokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

from transformers import BertTokenizer, BertModel
import joblib
import requests
# %%

def preprocess_with_bert(tokenizer, model, text, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    # use the output of the [CLS] token as the sentence embedding
    sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()
    return sentence_embedding

# calculate the accuracy
def acc(pred, label):
    pred = torch.round(pred)
    correct = (pred == label).float()
    acc = correct.sum() / correct.numel()
    return acc

# define the SimpleNN model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_prob=0.1):
        super(SimpleNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
# %%

def main():

    # %%
    # Datasets: go_emotions
    # https://www.kaggle.com/datasets/debarshichanda/goemotions/data

    df1 = pd.read_csv('dataset/goemotions_1.csv')
    df2 = pd.read_csv('dataset/goemotions_2.csv')
    df3 = pd.read_csv('dataset/goemotions_3.csv')

    df = pd.concat([df1, df2, df3])
    df = df.reset_index()
    df['index'] = [i for i in range(df.shape[0])]
    df = df.set_index('index')

    # %%
    # Preprocessing

    # use GPU(for Mac)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # use GPU(for CUDA)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # %%
    # use BERT to embed each text in the df['text'] column
    # This step may take a long time, depending on your hardware performance
    # I strongly recommend you to save the results to the local environment
    # If you have already run this step, you can skip it
    df['bert_embedding'] = df['text'].apply(preprocess_with_bert)

    # save the embeddings to the local environment
    joblib.dump(df['bert_embedding'].tolist(), 'bert_embeddings.joblib')

    # %%
    # load the embeddings from the local environment
    bert_embeddings = joblib.load('bert_embeddings.joblib')
    df['bert_embedding'] = bert_embeddings

    # extract the features and labels
    X = pd.DataFrame(df['bert_embedding'].to_list())
    y = df.iloc[:, 9:37].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # transform BERT embeddings to tensors(used in PyTorch)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    valid_data = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 100
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)

    # %%
    # input_dim = X_train.shape[1]  # BERT embedding size
    input_dim = 768 # 768 is the BERT embedding size
    hidden_dim = 64
    # output_dim = y_train.shape[1]
    output_dim = 28 # 28 is the number of emotion labels
    drop_prob = 0.25

    model = SimpleNN(input_dim, hidden_dim, output_dim, drop_prob).to(device)
    print(model)

    # %%
    # loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # %%
    # train the model
    epochs = 10
    clip = 5
    valid_loss_min = np.Inf

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(inputs)
            
            loss = criterion(output, labels)
            loss.backward()
            
            train_losses.append(loss.item())
            
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            accuracy = acc(output, labels)
            train_acc += accuracy
        
        model.eval()
        val_losses = []
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                output = model(inputs)
                
                val_loss = criterion(output, labels)
                val_losses.append(val_loss.item())
                
                accuracy = acc(output, labels)
                val_acc += accuracy
        
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc / len(train_loader)
        epoch_val_acc = val_acc / len(valid_loader)
        
        print(f'Epoch {epoch + 1}')
        print(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        print(f'Train Accuracy: {epoch_train_acc * 100:.2f}%, Val Accuracy: {epoch_val_acc * 100:.2f}%')
        
        if epoch_val_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {epoch_val_loss:.6f}). Saving model ...')
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = epoch_val_loss

    print('Training complete.')

if __name__ == '__main__':
    main()