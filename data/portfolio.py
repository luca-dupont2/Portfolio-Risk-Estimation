import yfinance as yf
from typing import List
import pandas as pd
import numpy as np
import streamlit as st

YEARLY_TRADING_DAYS = 252
PERIOD_MAP = {"3mo": 0.25, "6mo": 0.5, "1y": 1, "2y": 2, "5y": 5}

class Portfolio :
    def __init__(self, tickers: List[str], values: List[float], period: str) -> None:
        self.tickers = tickers
        self.values = values
        self.period = period
        self.data = None
        self.initial_portfolio_value = sum(values)

        self.log_returns = None
        self.log_std = None

        self.annualized_pct_returns = None
        self.annualized_pct_stds = None

        self.portfolio_return = None
        self.portfolio_std = None

    @st.cache_data
    def get_historical_data(self):
        """
        Fetch historical stock data from Yahoo Finance with caching.

        Args:
            tickers (Union[str, List[str]]): Ticker symbol(s) for the stock(s).
            period (str): Time period for the historical data (e.g., '1y', '2y').

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the closing prices of the stock(s), or None if an error occurs.

        Raises:
            ValueError: If no data is retrieved for the given tickers.
        """
        try:
            data: pd.DataFrame = yf.download(self.tickers, period=self.period)["Close"]
            if data.empty:
                raise ValueError(f"No data retrieved for tickers: {self.tickers}")
            self.data = data
        except Exception as e:
            print(f"Error gathering stock data: {e}")
            self.data = None

    def get_log_returns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns for a given DataFrame of stock prices.

        Args:
            data (pd.DataFrame): DataFrame containing stock prices.

        Returns:
            pd.DataFrame: DataFrame containing the log returns.
        """
        return np.log(data / data.shift(1)).dropna()
    
    def get_annualized_pct_returns(self):
        """
        Calculate the annualized percentage returns for the portfolio.

        Returns:
            np.ndarray: Array containing the annualized percentage returns.
        """
        self.log_returns = self.get_log_returns(self.data)

        annualized_pct_returns = self.log_returns.mean() * YEARLY_TRADING_DAYS

        self.annualized_pct_returns = annualized_pct_returns

    def get_annualized_log_std(self):
        """
        Calculate the annualized log standard deviation for the portfolio.

        Returns:
            np.ndarray: Array containing the annualized log standard deviations.
        """

        annualized_log_stds = self.log_returns.std() * np.sqrt(YEARLY_TRADING_DAYS)

        self.log_std = annualized_log_stds
    
    def log_std_to_pct_std(self):
        """
        Convert log standard deviation to percentage standard deviation.

        Args:
            log_std (float): Log standard deviation.

        Returns:
            float: Percentage standard deviation.
        """
        if np.any(self.log_std == 0):
            return 0.0  # No volatility means no percentage std deviation
        
        self.annualized_pct_stds = np.sqrt(np.exp(self.log_std**2) - 1)

    def get_portfolio_attributes(self) :
        """
        Calculate the portfolio return and standard deviation.

        Returns:
            Tuple[float, float]: Portfolio return and standard deviation.
        """
        self.portfolio_return = np.average(self.annualized_pct_returns, weights=self.values)
        self.portfolio_std = np.average(self.annualized_pct_stds, weights=self.values)