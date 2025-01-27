import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import ta
import warnings
from dotenv import load_dotenv
load_dotenv()
# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure PyTorch (add these lines)
import torch
torch.set_num_threads(1)  # Limit number of threads
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear CUDA cache
# Import using the package structure
from scripts.price_predictor import StockPredictor
from scripts.portfolio_optimizer import PortfolioOptimizer
from scripts.sentiment_analyzer import SentimentAnalyzer

# Page configuration
st.set_page_config(
    page_title="AI Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px 16px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Technical Analysis Class
class TechnicalAnalysis:
    def __init__(self, data):
        self.data = data
        
    def calculate_all_indicators(self):
        # RSI
        self.data['RSI'] = ta.momentum.RSIIndicator(self.data['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(self.data['Close'])
        self.data['BB_Upper'] = bollinger.bollinger_hband()
        self.data['BB_Lower'] = bollinger.bollinger_lband()
        self.data['BB_Middle'] = bollinger.bollinger_mavg()
        
        return self.data

# Risk Analysis Class
class RiskAnalysis:
    def __init__(self, data):
        self.data = data
        
    def calculate_metrics(self):
        returns = self.data['Close'].pct_change()
        metrics = {
            'volatility': returns.std() * np.sqrt(252),
            'var_95': np.percentile(returns.dropna(), 5),
            'max_drawdown': self.calculate_max_drawdown(),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns)
        }
        return metrics
    
    def calculate_max_drawdown(self):
        peak = self.data['Close'].expanding(min_periods=1).max()
        drawdown = (self.data['Close'] - peak) / peak
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.01):
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        return yf.download(ticker, start=start_date, end=end_date)
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

@st.cache_data
def load_multiple_stocks(tickers, start_date, end_date):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = {ticker: executor.submit(load_data, ticker, start_date, end_date) 
                  for ticker in tickers}
    return {ticker: future.result() for ticker, future in results.items()}

def format_sentiment(sentiment_label, score):
    if sentiment_label == 'POSITIVE':
        color = 'green'
    elif sentiment_label == 'NEGATIVE':
        color = 'red'
    else:
        color = 'gray'
    return f"<span style='color: {color}'>{sentiment_label} ({score:.2f})</span>"

def create_price_chart(data, ticker):
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=ticker
    ))
    
    # Add moving averages
    ma20 = data['Close'].rolling(window=20).mean()
    ma50 = data['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(x=data.index, y=ma20, name='20 Day MA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=ma50, name='50 Day MA', line=dict(color='blue')))
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_white',
        height=600
    )
    
    return fig

def create_technical_chart(data, ticker, indicators):
    fig = go.Figure()
    
    # Main price chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=ticker
    ))
    
    # Add technical indicators
    if 'Bollinger Bands' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], 
                               name='BB Upper', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], 
                               name='BB Lower', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], 
                               name='BB Middle', line=dict(color='gray')))
    
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_white',
        height=600
    )
    
    return fig

def create_risk_metrics_display(risk_metrics):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Volatility (Annual)", f"{risk_metrics['volatility']:.2%}")
    with col2:
        st.metric("Value at Risk (95%)", f"{risk_metrics['var_95']:.2%}")
    with col3:
        st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
    with col4:
        st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")

def main():
    st.title("AI Financial Dashboard 📈")
    
    # Enhanced Sidebar
    st.sidebar.header("Dashboard Settings")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
    days_to_analyze = st.sidebar.slider("Days to Analyze", 30, 365, 180)
    prediction_days = st.sidebar.slider("Prediction Days", 1, 30, 7)
    
    # Technical Analysis Settings
    show_technical = st.sidebar.checkbox("Show Technical Analysis", True)
    technical_indicators = st.sidebar.multiselect(
        "Technical Indicators",
        ["RSI", "MACD", "Bollinger Bands"],
        default=["Bollinger Bands"]
    )
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_analyze)
    
    # Load data
    with st.spinner('Loading stock data...'):
        data = load_data(ticker, start_date, end_date)
        
    if data is None or data.empty:
        st.error("No data available for the selected ticker.")
        return
        
    # Calculate technical indicators
    if show_technical:
        ta_analyzer = TechnicalAnalysis(data.copy())
        data = ta_analyzer.calculate_all_indicators()
    
    # Calculate risk metrics
    risk_analyzer = RiskAnalysis(data)
    risk_metrics = risk_analyzer.calculate_metrics()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price Analysis", 
        "Technical Analysis", 
        "Risk Analysis",
        "Price Prediction",
        "Portfolio & Sentiment"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Price Analysis")
            price_chart = create_price_chart(data, ticker)
            st.plotly_chart(price_chart, use_container_width=True)
        
        with col2:
            st.subheader("Stock Statistics")
            current_price = data['Close'][-1]
            price_change = data['Close'][-1] - data['Close'][-2]
            price_change_pct = (price_change / data['Close'][-2]) * 100
            
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change_pct:.2f}%"
            )
            
            vol = data['Volume'][-1]
            avg_vol = data['Volume'].mean()
            high_52w = data['High'].rolling(window=252).max()[-1]
            low_52w = data['Low'].rolling(window=252).min()[-1]
            
            st.metric("52 Week High", f"${high_52w:.2f}")
            st.metric("52 Week Low", f"${low_52w:.2f}")
            st.metric("Volume", f"{vol:,.0f}", f"{((vol/avg_vol)-1)*100:.1f}% avg")
    
    with tab2:
        if show_technical:
            st.subheader("Technical Analysis")
            technical_chart = create_technical_chart(data, ticker, technical_indicators)
            st.plotly_chart(technical_chart, use_container_width=True)
            
            if "RSI" in technical_indicators:
                st.subheader("Relative Strength Index (RSI)")
                st.line_chart(data['RSI'])
            
            if "MACD" in technical_indicators:
                st.subheader("MACD")
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
                macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal'))
                st.plotly_chart(macd_fig)
    
    with tab3:
        st.subheader("Risk Analysis")
        create_risk_metrics_display(risk_metrics)
        
        # Additional risk visualizations
        st.subheader("Returns Distribution")
        returns = data['Close'].pct_change().dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns'))
        fig.update_layout(title="Daily Returns Distribution")
        st.plotly_chart(fig)
    
    with tab4:
        st.subheader("Price Prediction")
        try:
            with st.spinner('Calculating price predictions...'):
                predictor = StockPredictor()
                X, y = predictor.prepare_data(data)
                predictor.build_model(60)
                
                # Add progress bar for training
                progress_bar = st.progress(0)
                epochs = 10
                
                # Custom training loop with progress bar
                predictor.model.train()
                for epoch in range(epochs):
                    predictor.optimizer.zero_grad()
                    output = predictor.model(X)
                    loss = predictor.criterion(output, y)
                    loss.backward()
                    predictor.optimizer.step()
                    
                    # Update progress bar
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                
                # Make prediction
                prediction = predictor.predict(X[-1])
                
                st.success(f"Predicted price (next day): ${prediction[0][0]:.2f}")
                
                # Show prediction confidence interval
                confidence = 0.95
                std_dev = data['Close'].std()
                margin = std_dev * 1.96
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Upper Bound", f"${(prediction[0][0] + margin):.2f}")
                with col2:
                    st.metric("Lower Bound", f"${(prediction[0][0] - margin):.2f}")
                
        except Exception as e:
            st.error(f"Error in price prediction: {str(e)}")
    
    with tab5:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Optimization")
            portfolio_tickers = st.text_input(
                "Enter ticker symbols (comma-separated)",
                "AAPL,MSFT,GOOGL,AMZN"
            ).split(',')
            
            if portfolio_tickers:
                try:
                    with st.spinner('Optimizing portfolio...'):
                        portfolio_data = pd.DataFrame()
                        for tick in portfolio_tickers:
                            stock_data = load_data(tick.strip(), start_date, end_date)
                            if stock_data is not None:
                                portfolio_data[tick] = stock_data['Close'].pct_change()
                        
                        if not portfolio_data.empty:
                            optimizer = PortfolioOptimizer()
                            weights = optimizer.optimize_portfolio(portfolio_data.dropna())
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=portfolio_tickers,
                                values=weights,
                                textinfo='label+percent',
                                hole=.3
                            )])
                            fig.update_layout(title="Optimal Portfolio Allocation")
                            st.plotly_chart(fig)
                            
                            weight_df = pd.DataFrame({
                                'Stock': portfolio_tickers,
                                'Weight': [f"{w:.2%}" for w in weights]
                            })
                            st.table(weight_df)
                except Exception as e:
                    st.error(f"Error in portfolio optimization: {str(e)}")
        
        with col2:
            st.subheader("Sentiment Analysis")
            NEWS_API_KEY = os.getenv('NEWS_API_KEY')
            
            try:
                with st.spinner('Analyzing news sentiment...'):
                    analyzer = SentimentAnalyzer(NEWS_API_KEY)
                    sentiments = analyzer.get_company_sentiment(ticker)
                    
                    if sentiments:
                        avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
                        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
                        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'NEGATIVE')
                        
                        st.metric("Average Sentiment Score", f"{avg_score:.2f}")
                        
                        fig = go.Figure(data=[go.Bar(
                            x=['Positive', 'Negative'],
                            y=[positive_count, negative_count],
                            marker_color=['green', 'red']
                        )])
                        fig.update_layout(title="Sentiment Distribution")
                        st.plotly_chart(fig)
                        
                        for sentiment in sentiments:
                            with st.expander(sentiment['title']):
                                st.markdown(f"**Sentiment:** {format_sentiment(sentiment['sentiment'], sentiment['score'])}", unsafe_allow_html=True)
                                st.markdown(f"**Published:** {sentiment['published_at']}")
                                if sentiment['url']:
                                    st.markdown(f"[Read Article]({sentiment['url']})")
                    else:
                        st.info("No recent news articles found for this company.")
                        
            except Exception as e:
                st.error(f"Error in sentiment analysis: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Built with Streamlit • Data from Yahoo Finance & News API</p>
            <p>This is a demo application. Do not use for actual trading decisions.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()