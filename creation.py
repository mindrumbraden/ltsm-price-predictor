#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:42:16 2025

@author: bradenmindrum
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

#%%

def clean_df(df):
    dropped_columns = ["name", "volume", "marketCap", "timestamp"]
    dt_columns = ["timeOpen", "timeClose", "timeHigh", "timeLow"]
    dates = "timeOpen"
    df.drop(columns=dropped_columns, inplace=True)
    df[dt_columns] = df[dt_columns].apply(pd.to_datetime)
    df.insert(0, column="date", value=df[dates].dt.date.astype("datetime64[ns]"))
    
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

#%%

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        '''
        ***Explain*** Why do we need h_0 and c_0?
        '''
        
        #ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        ula, (h_out, _) = self.lstm(x)              # If (h_0, c_0) not provided
                                                    # then initialized to 0 anyway.
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
    
#%%
    
df = pd.read_csv("bitcoin.csv", sep=";", index_col=False)
clean_df(df)
print(df)

#%%

# Let's start by doing everything with the close end price for each day
# i.e., use index 8
data = df.iloc[:,8]
data = data[::-1].values
data = data.reshape(-1, 1)
plt.plot(data)

#%%

scale = MinMaxScaler()
data = scale.fit_transform(data)
plt.plot(data)

#%%

window = 10
x, y = sliding_windows(data, 10)

#%%

val_size = 30
test_size = 30
train_size = len(y) - val_size - test_size

train_x = Variable(torch.Tensor(np.array(x[0:train_size])))
train_y = Variable(torch.Tensor(np.array(y[0:train_size])))

val_x = Variable(torch.Tensor(np.array(x[train_size:train_size+val_size])))
val_y = Variable(torch.Tensor(np.array(y[train_size:train_size+val_size])))

test_x = Variable(torch.Tensor(np.array(x[train_size+val_size:])))
test_y = Variable(torch.Tensor(np.array(y[train_size+val_size:])))

#%%

num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1
seq_length = window

lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

#%%

criterion = torch.nn.MSELoss()    
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(train_x)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
     
#%%

lstm.eval()
val_predict = lstm(val_x)

data_predict = val_predict.data.numpy()
data_predict = scale.inverse_transform(data_predict)

data_actual = val_y.data.numpy()
data_actual = scale.inverse_transform(data_actual)

plt.plot(data_predict)
plt.plot(data_actual)
plt.show()

#%%

"""
Need to transfer these items to main.py
"""



























































