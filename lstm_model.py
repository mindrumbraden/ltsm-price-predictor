#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 14:02:23 2025

@author: bradenmindrum
"""

#%%

import torch
import torch.nn as nn

#%%

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=2, num_layers=1):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (out, _) = self.lstm(x)
        out = out.view(-1, self.hidden_size)
        out = self.fully_connected(out)
        return out
    
#%%