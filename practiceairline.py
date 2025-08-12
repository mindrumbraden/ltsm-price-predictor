#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:28:19 2025

@author: bradenmindrum
"""

#%%

'''
We are learning how to use LSTM to do forecasting. Follow the code and explain it when it is needed.
'''
#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

#%%

#load the dataset

training_set = pd.read_csv('airline-passengers.csv')

training_set = training_set.iloc[:,1:2].values

plt.plot(training_set, label = 'Airlines Data')
plt.show()

#%%

'''
***Explain*** What is the sliding_windows function doing here?, and what does the seq_length value mean?
'''

seq_length = 5

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)



sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

x, y = sliding_windows(training_set, seq_length)

x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

#%%

'''
LSTM Network
'''

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
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

num_epochs = 100
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1

'''
***Explain*** Why are the input size and the num_classes equal to one?, should not it be input_size = seq_length? 
'''

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

# for np in lstm.parameters():
#     print(np)

#%%

criterion = torch.nn.MSELoss()    
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
      
#%%

lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

#Plot the Predictions
plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()

#%%

'''
***Explain*** Rerun your model with 5000 epochs. Do you notice something strange? If yes, provide an explanation.
'''

num_epochs = 5000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

# for np in lstm.parameters():
#     print(np)


criterion = torch.nn.MSELoss()    
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
     
lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

#Plot the Predictions
plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()

#%%

'''
Implement MSE calculation for the test data
'''
# First, reset to 2000 epochs.

num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

#Plot the Predictions
plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()

#%%

criterion = torch.nn.MSELoss()  

lstm.eval()
test_predictions = lstm(testX)
print("Testing Error:", criterion(test_predictions, testY).item())

#%%

def mse_error(pred, actual):
    squared_diffs = (pred - actual)**2
    mse = torch.mean(squared_diffs).item()
    #print("Test MSE Error:", mse)
    return mse

mse_error(test_predictions, testY)

#%%
        
# Grid search

seq_lengths = [3, 5, 10]
hidden_sizes = [2, 5, 10]

num_epochs = 2000
learning_rate = 0.01

input_size = 1
num_layers = 1
num_classes = 1

for seq_length in seq_lengths:
    for hidden_size in hidden_sizes:
        # Get Data as before
        x, y = sliding_windows(training_data, seq_length)
        train_size = int(len(y) * 0.67)
        test_size = len(y) - train_size
        dataX = Variable(torch.Tensor(np.array(x)))
        dataY = Variable(torch.Tensor(np.array(y)))
        trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
        trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
        testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
        testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
        
        # Define and train LSTM
        lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
        criterion = torch.nn.MSELoss()    
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            outputs = lstm(trainX)
            optimizer.zero_grad()
            loss = criterion(outputs, trainY)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch:", epoch, "Loss:", loss.item())
        
        # Obtain test error & create title
        lstm.eval()
        test_predictions = lstm(testX)
        mse = mse_error(test_predictions, testY)
        title = str("seq_length = " + str(seq_length) + " & hidden_size = "
                    + str(hidden_size) + "\n" + "Test MSE Error: " + str(mse))
        
        # Print plot with test error and fancy
        # lstm.eval() (above)
        train_predict = lstm(dataX)
        data_predict = train_predict.data.numpy()
        dataY_plot = dataY.data.numpy()
        data_predict = sc.inverse_transform(data_predict)
        dataY_plot = sc.inverse_transform(dataY_plot)
        plt.axvline(x=train_size, c='r', linestyle='--')
        plt.plot(dataY_plot)
        plt.plot(data_predict)
        #plt.suptitle("Time-Series Prediction")
        plt.title(title)
        plt.show()
    




















































