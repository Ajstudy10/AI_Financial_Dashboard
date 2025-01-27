class TechnicalAnalysis:
    def __init__(self, data):
        self.data = data
        
    def calculate_rsi(self, period=14):
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        exp1 = self.data['Close'].ewm(span=fast).mean()
        exp2 = self.data['Close'].ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
        
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        sma = self.data['Close'].rolling(window=period).mean()
        rolling_std = self.data['Close'].rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, sma, lower_band
