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
        self.log_stds = None

        self.annualized_pct_returns = None
        self.annualized_pct_stds = None

        self.portfolio_return = None
        self.portfolio_std = None

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

    def get_log_returns(self) -> pd.DataFrame:
        """
        Calculate log returns for a given DataFrame of stock prices.

        Args:
            data (pd.DataFrame): DataFrame containing stock prices.

        Returns:
            pd.DataFrame: DataFrame containing the log returns.
        """
        return np.log(self.data / self.data.shift(1)).dropna()
    
    def get_annualized_pct_returns(self):
        """
        Calculate the annualized percentage returns for the portfolio.

        Returns:
            np.ndarray: Array containing the annualized percentage returns.
        """
        self.log_returns = self.get_log_returns()

        annualized_pct_returns = self.log_returns.mean() * YEARLY_TRADING_DAYS

        self.annualized_pct_returns = annualized_pct_returns

    def get_log_stds(self) -> pd.DataFrame:
        """
        Calculate the annualized log standard deviation for the portfolio.

        Returns:
            np.ndarray: Array containing the annualized log standard deviations.
        """

        return self.log_returns.std()
    
    def get_annualized_pct_stds(self):
        """
        Convert log standard deviation to percentage standard deviation.

        Args:
            log_std (float): Log standard deviation.

        Returns:
            float: Percentage standard deviation.
        """
        
        self.log_stds = self.get_log_stds()
        
        annualized_log_stds = self.log_stds * np.sqrt(YEARLY_TRADING_DAYS)
        
        self.annualized_pct_stds = np.sqrt(np.exp(annualized_log_stds**2) - 1)

    def get_portfolio_attributes(self) :
        """
        Calculate the portfolio return and standard deviation.

        Returns:
            Tuple[float, float]: Portfolio return and standard deviation.
        """

        self.get_annualized_pct_returns()
        self.get_annualized_pct_stds()

        self.portfolio_return = np.average(self.annualized_pct_returns, weights=self.values)
        self.portfolio_std = np.average(self.annualized_pct_stds, weights=self.values)

    def get_equity_part(self, ticker_df : pd.DataFrame):
        """
        Calculate the equity part of the portfolio.

        Args:
            ticker_df (pd.DataFrame): DataFrame containing the tickers and contract types.

        Returns:
            float: Equity part of the portfolio.
        """

        equity_part = 0

        contract_types = ticker_df.loc[ticker_df["Symbol"].isin(self.tickers), ["Symbol", "Contract Type"]]

        for ticker in self.tickers:
            contract_type = contract_types.loc[contract_types["Symbol"] == ticker, "Contract Type"].values[0]
            if contract_type == "Equity" :
                equity_part += self.values[self.tickers.index(ticker)]

        return equity_part / self.initial_portfolio_value