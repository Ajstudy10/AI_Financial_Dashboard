class RiskManager:
    def calculate_position_size(self, account_size, risk_per_trade, stop_loss):
        risk_amount = account_size * (risk_per_trade / 100)
        position_size = risk_amount / stop_loss
        return position_size
        
    def calculate_kelly_criterion(self, win_rate, win_loss_ratio):
        return (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio