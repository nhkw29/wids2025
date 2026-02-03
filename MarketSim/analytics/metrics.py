import pandas as pd
import numpy as np
from .tape import Tape

class MarketMetrics:
    def __init__(self, data_source):
        if hasattr(data_source, 'to_dataframe'):
            self.df = data_source.to_dataframe()
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source
        else:
            self.df = pd.DataFrame()

    def compute_vwap(self):
        if self.df.empty:
            return None
        if 'price' not in self.df.columns or 'qty' not in self.df.columns:
            return 0.0
            
        total_dollar_volume = (self.df['price'] * self.df['qty']).sum()
        total_volume = self.df['qty'].sum()
        
        if total_volume == 0:
            return 0.0
            
        vwap = total_dollar_volume / total_volume
        return vwap

    def calculate_vwap(self):
        return self.compute_vwap()

    def get_session_volatility(self):
        if self.df.empty:
            return 0.0
            
        price_col = 'mid_price' if 'mid_price' in self.df.columns else 'price'
        
        if price_col not in self.df.columns:
            return 0.0

        if isinstance(self.df.index, pd.DatetimeIndex):
            resampled = self.df[price_col].resample('1s').last().dropna()
        else:
            resampled = self.df[price_col]

        if len(resampled) < 2:
            return 0.0
            
        log_returns = np.log(resampled / resampled.shift(1)).dropna()
        volatility = log_returns.std()
        return volatility

    def calculate_session_volatility(self):
        return self.get_session_volatility()
    
    def get_rolling_volatility(self, window_size=60):
        if self.df.empty or 'mid_price' not in self.df.columns:
            return None
        
        resampled_mid = self.df['mid_price'].resample('1s').last().dropna()
        log_returns = np.log(resampled_mid / resampled_mid.shift(1)).dropna()
        rolling_volatility = log_returns.rolling(window=window_size).std()
        return rolling_volatility