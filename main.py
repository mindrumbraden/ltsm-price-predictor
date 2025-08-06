#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:03:35 2025

@author: bradenmindrum
"""

import pandas as pd

def clean_df(df):
    dropped_columns = ["name", "volume", "marketCap", "timestamp"]
    dt_columns = ["timeOpen", "timeClose", "timeHigh", "timeLow"]
    dates = "timeOpen"
    df.drop(columns=dropped_columns, inplace=True)
    df[dt_columns] = df[dt_columns].apply(pd.to_datetime)
    df.insert(0, column="date", value=df[dates].dt.date.astype('datetime64'))
    

def main():
    try:
        df = pd.read_csv("bitcoin.csv", sep=";")
    except:
        # https://coinmarketcap.com/currencies/bitcoin/historical-data/
        print("No file 'bitcoin.csv'.")
    try:
        clean_df(df)
    except:
        print("Error in function 'clean_df'")
        return False
    print(df)
    
    
    

if __name__ == "__main__":
    main()
    