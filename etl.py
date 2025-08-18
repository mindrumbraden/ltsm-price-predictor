#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:48:37 2025

@author: bradenmindrum
"""

#%%

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.autograd import Variable

#%%

def delete_columns_not_in_list(df, columns_to_keep):
    dropped_columns = [col for col in df.columns if col not in columns_to_keep]
    df.drop(columns=dropped_columns, inplace=True)
    
def 
    
    

def clean_df(df):
    """

    Parameters
    ----------
    df : pandas dataframe
        Assumes the dataframe was read as pd.read_csv("bitcoin.csv", sep=";")
        
        Drops the columns "name", "volume", "marketCap", and "timestamp"; 
        
        Changes the columns "timeOpen", "timeClose", "timeHigh", and "timeLow"
        to dtype datetime
        
        Adds a new column "date" that is purely a date

    Returns
    -------
    None. Modifies the original dataframe

    """
    dropped_columns = ["name", "volume", "marketCap", "timestamp"]
    dt_columns = ["timeOpen", "timeClose", "timeHigh", "timeLow"]
    dates = "timeOpen"
    df.drop(columns=dropped_columns, inplace=True)
    df[dt_columns] = df[dt_columns].apply(pd.to_datetime)
    df.insert(0, column="date", value=df[dates].dt.date.astype('datetime64[ns]'))

def export_transform_load(file="bitcoin.csv"):
    """

    Parameters
    ----------
    file : str, optional
        The file path to the associated data

    Returns
    -------
    df : pandas dataframe
        Already cleaned data frame according to the function clean_df.

    """
    df = pd.read_csv(file, sep=";")
    clean_df(df)
    return df

#%%

def sliding_windows(data, seq_length=10):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

def shape_and_transform_data(df, transform=MinMaxScaler()):
    data = df.iloc[:,8]
    data = data[::-1].values
    data = data.reshape(-1, 1)
    data = transform.fit_transform(data)
    return data, transform

def obtain_x_y(df):
    """

    Parameters
    ----------
    df : pandas dataframe
        As created by export_transform_load(file="bitcoin.csv")

    Returns
    -------
    x : np array
        X
        
    y : np array
        Y

    """
    data, transform = shape_and_transform_data(df)
    x, y = sliding_windows(data)
    return x, y, transform

#%%

def train_val_test_split(x, y, val_size=30, test_size=30):
    train_size = len(y) - val_size - test_size
    
    train_x = Variable(Tensor(np.array(x[0:train_size])))
    train_y = Variable(Tensor(np.array(y[0:train_size])))
    
    val_x = Variable(Tensor(np.array(x[train_size:train_size+val_size])))
    val_y = Variable(Tensor(np.array(y[train_size:train_size+val_size])))
    
    test_x = Variable(Tensor(np.array(x[train_size+val_size:])))
    test_y = Variable(Tensor(np.array(y[train_size+val_size:])))
    
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)