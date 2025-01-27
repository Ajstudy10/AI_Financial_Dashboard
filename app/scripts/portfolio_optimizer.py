import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self):
        self.returns = None
        self.weights = None
        
    def calculate_portfolio_metrics(self, weights):
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(self.returns.cov() * 252, weights))
        )
        return portfolio_return, portfolio_std
        
    def optimize_portfolio(self, returns_data, risk_free_rate=0.01):
        self.returns = returns_data
        
        def negative_sharpe_ratio(weights):
            portfolio_return, portfolio_std = self.calculate_portfolio_metrics(weights)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe_ratio
        
        num_assets = len(self.returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        result = minimize(
            negative_sharpe_ratio,
            num_assets * [1./num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.weights = result.x
        return self.weights