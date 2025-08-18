#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:03:35 2025

@author: bradenmindrum
"""

#%%

# Modules used, but not used within the main.py scope
# import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np


from etl import export_transform_load, obtain_x_y, train_val_test_split
from lstm_model import LSTM

#%%

    
#%%

def main():
    df = export_transform_load("bitcoin.csv")
    
    # 0 corresponds to date. 8 corresponds to closing price
    plt.plot(df.iloc[:,0], df.iloc[:,8])
    plt.title("Bitcoin Closing Price Over Time")
    plt.show()

    
    
    
    
    
    
    
    
    
"""    
    # Create usable data
    x, y, transform = obtain_x_y(df)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = train_val_test_split(x, y)
    
    # Initialize model
    lstm = LSTM()
    
    # Set parameters, optimizer, and criterion
    num_epochs = 500
    learning_rate = 0.01
    optimizer = Adam(lstm.parameters(), lr=learning_rate)
    criterion = MSELoss()  
    
    # Train model
    for epoch in range(num_epochs):
        outputs = lstm(train_x)
        optimizer.zero_grad()
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    lstm.eval()
    val_predict = lstm(val_x)

    data_predict = val_predict.data.numpy()
    data_predict = transform.inverse_transform(data_predict)

    data_actual = val_y.data.numpy()
    data_actual = transform.inverse_transform(data_actual)

    shifted_data = np.insert(data_actual[:-1], 0, np.nan)

    plt.plot(shifted_data)
    plt.plot(data_predict)
    plt.show()
    
    return 0
"""

#%% 

if __name__ == "__main__":
    main()
    
#%% 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    