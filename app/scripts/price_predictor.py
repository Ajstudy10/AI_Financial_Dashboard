# scripts/price_predictor.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, data, lookback=60):
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1,1))
        X, y = [], []
        
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X).reshape(-1, lookback, 1).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        return X, y
    
    def build_model(self, lookback):
        self.model = LSTMModel(
            input_dim=1,
            hidden_dim=50,
            num_layers=2,
            output_dim=1
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def train(self, X, y, epochs=50, batch_size=32):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).reshape(-1, X.shape[0], 1).to(self.device)
            predictions = self.model(X)
            predictions = predictions.cpu().numpy()
            return self.scaler.inverse_transform(predictions.reshape(-1, 1))