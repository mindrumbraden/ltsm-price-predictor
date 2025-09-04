import numpy as np
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_cleaning import clean_df
from utils.preprocessing import sliding_windows
from models.lstm_model import LSTM

def train_model(csv_path: str, seperator: str, target_column: str, 
                window: int = 10, 
                val_size: int = 30, test_size: int = 30, 
                num_epochs: int = 2000, learning_rate: float = 0.01,
                hidden_size: int = 2):
    """
    Train LSTM model on given CSV file.
    """
    # Load and clean data
    df = pd.read_csv(csv_path, sep=seperator, index_col=False)
    df = clean_df(df, target_col=target_column)

    # Extract target series
    data = df[target_column][::-1].values.reshape(-1, 1)

    # Normalize
    scale = MinMaxScaler()
    data = scale.fit_transform(data)

    # Sliding window
    x, y = sliding_windows(data, window)

    # Train/Val/Test Split
    train_size = len(y) - val_size - test_size
    train_x = Variable(torch.Tensor(np.array(x[0:train_size])))
    train_y = Variable(torch.Tensor(np.array(y[0:train_size])))

    val_x = Variable(torch.Tensor(np.array(x[train_size:train_size+val_size])))
    val_y = Variable(torch.Tensor(np.array(y[train_size:train_size+val_size])))

    # Model parameters
    lstm = LSTM(num_classes=1, input_size=1, hidden_size=hidden_size, num_layers=1, seq_length=window)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        outputs = lstm(train_x)
        optimizer.zero_grad()
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            
    print("Training Complete.")
    print("Now Showing Results.")
    
    # Validation
    lstm.eval()
    val_predict = lstm(val_x).data.numpy()
    data_predict = scale.inverse_transform(val_predict)
    data_actual = scale.inverse_transform(val_y.data.numpy())

    plt.plot(df['date'].iloc[train_size:train_size+val_size], data_predict, label="Predicted")
    plt.plot(df['date'].iloc[train_size:train_size+val_size], data_actual, label="Actual")
    plt.title(f"Validation Results ({target_column})")
    plt.legend()
    plt.show()

    return lstm
