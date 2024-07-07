import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['종가'] = data['종가'].str.replace(',', '').astype(float)
    data['날짜'] = pd.to_datetime(data['날짜'].str.replace(' ', ''))
    
    close_prices = data['종가'].values.reshape(-1, 1)
    dates = data['날짜'].values
    
    scaler = MinMaxScaler()
    close_prices_normalized = scaler.fit_transform(close_prices)
    
    return close_prices, close_prices_normalized, dates, scaler

# 시퀀스 데이터 생성 함수
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# LSTM 모델 클래스 정의
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 모델 학습 함수
def train_model(model, X_train, y_train, X_val, y_val, num_epochs=50, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses

# 예측 함수
def make_predictions(model, X, scaler, seq_length, dates, close_prices_normalized, end_date_str='2025-01-01'):
    model.eval()
    with torch.no_grad():
        predicted = model(X).detach().numpy()
        predicted = scaler.inverse_transform(predicted)
    
    date_diff = (dates[1] - dates[0]).astype('timedelta64[D]').astype(int)
    end_date = pd.Timestamp(end_date_str)
    future_steps = max((end_date - pd.Timestamp(dates[-1])).days // date_diff, 0)
    
    future_predictions = []
    input_seq = close_prices_normalized[-seq_length:].reshape(1, seq_length, 1)
    
    if future_steps > 0:
        for _ in range(future_steps):
            input_tensor = torch.from_numpy(input_seq).float()
            with torch.no_grad():
                pred = model(input_tensor)
                future_predictions.append(pred.item())
                input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(start=dates[-1], periods=future_steps + 1, freq='D')[1:]
    else:
        future_predictions, future_dates = [], []
    
    return predicted, future_predictions, future_dates

# 시각화 함수
def plot_results(dates, close_prices, predicted, future_dates, future_predictions, train_losses, val_losses, seq_length):
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates, close_prices, label='True')
    plt.plot(dates[seq_length:], predicted, label='Predicted')
    if future_dates and future_predictions:
        plt.plot(future_dates, future_predictions, label='Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 주요 실행 흐름
file_path =  '경로'
close_prices, close_prices_normalized, dates, scaler = load_and_preprocess_data(file_path)

seq_length = 50
X, y = create_sequences(close_prices_normalized, seq_length)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

input_size = 1
hidden_size = 50
output_size = 1
model = LSTM(input_size, hidden_size, output_size)

train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, num_epochs=50)

predicted, future_predictions, future_dates = make_predictions(
    model, X, scaler, seq_length, dates, close_prices_normalized
)

plot_results(dates, close_prices, predicted, future_dates, future_predictions, train_losses, val_losses, seq_length)
