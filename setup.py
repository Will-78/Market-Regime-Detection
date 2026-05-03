import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled

def setup():
    datasets = [
        'SPY',
        'QQQ',
        'IWM',
        'TLT',
        'GLD',
        'XLE',
        'XLF',
        'XLK',
        'SHY'
    ]

    df = yf.download(datasets, start='2020-01-01', end='2022-06-30')

    # work only with close prices for all tickers
    close = df['Close'].squeeze()

    returns = close.pct_change()
    returns = returns.dropna()

    print(returns.head())

    features = pd.DataFrame(index=returns.index)

    rolling_window = 20

    features = pd.DataFrame(index=returns.index)

    features['volatility'] = returns.rolling(rolling_window).std().mean(axis=1)

    features['avg_return'] = returns.rolling(rolling_window).mean().mean(axis=1)

    features['dispersion'] = returns.std(axis=1)

    features['correlation'] = returns.rolling(rolling_window).corr().groupby(level=0).mean().mean(axis=1)

    features['equity_bond_spread'] = (
        returns[['SPY','QQQ','IWM']].mean(axis=1) -
        returns['TLT']
    )

    features['drawdown'] = (close / close.rolling(rolling_window).max() - 1).mean(axis=1)

    features['vol_change'] = features['volatility'].diff()

    features = features.dropna()

    return features