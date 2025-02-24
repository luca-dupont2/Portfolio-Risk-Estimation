import numpy as np
import yfinance as yf
import pandas as pd
import socket
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Tuple, Union

# Load environment variables from .env file
load_dotenv()

# Constants
PERIOD_MAP = {"3mo": 0.25, "6mo": 0.5, "1y": 1, "2y": 2, "5y": 5}
FRED_API_KEY = os.getenv("FRED_API_KEY")


def check_wifi_connection() -> bool:
    """
    Check if there is an active internet connection by attempting to connect to Google DNS.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """

    try:
        # Try to connect to a well-known site (Google DNS) to check if there's an internet connection
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def check_api_key() -> bool :
    return os.path.isfile(".env")


def get_monthly_bank_rates_fred(years: int) -> pd.DataFrame:
    """
    Fetch monthly bank rates from the FRED API for a specified number of years.

    Args:
        years (int): Number of years of data to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing the date and corresponding bank rates.

    Raises:
        ValueError: If the FRED API request fails.
    """
    now = datetime.now()
    then = (now - relativedelta(years=years)).strftime("%Y-%m-%d")
    now_str = now.strftime("%Y-%m-%d")

    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={FRED_API_KEY}&file_type=json&observation_start={then}&observation_end={now_str}"

    response = requests.get(url)
    status_code = response.status_code
    if status_code == 200:
        data = response.json()
        observations = data["observations"]

        df = pd.DataFrame(observations)[["date", "value"]]
        return df

    else:
        raise ValueError(
            f"Error loading FRED interest bank data status code {status_code}"
        )


def get_historical_data(
    tickers: Union[str, List[str]], period: str
) -> Optional[pd.DataFrame]:
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
        data: pd.DataFrame = yf.download(tickers, period=period)["Close"]
        if data.empty:
            raise ValueError(f"No data retrieved for tickers: {tickers}")
        return data
    except Exception as e:
        print(f"Error gathering stock data: {e}")
        return None


def get_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns for a given DataFrame of stock prices.

    Args:
        data (pd.DataFrame): DataFrame containing stock prices.

    Returns:
        pd.DataFrame: DataFrame containing the log returns.
    """
    return np.log(data / data.shift(1)).dropna()


def get_annualized_pct_returns(
    log_returns: pd.DataFrame, period: Union[str, int, float]
) -> np.ndarray:
    """
    Calculate annualized percentage returns using log returns.

    Args:
        log_returns (pd.DataFrame): DataFrame containing log returns.
        period (Union[str, int, float]): Time period for annualization.

    Returns:
        np.ndarray: Array of annualized percentage returns.

    Raises:
        ValueError: If the period is invalid.
    """
    if isinstance(period, str) and period in PERIOD_MAP:
        period = PERIOD_MAP[period]
    elif not isinstance(period, (int, float)) or period <= 0:
        raise ValueError(
            f"Invalid period '{period}'. Must be positive or in {list(PERIOD_MAP.keys())}."
        )

    annualized_log_returns = log_returns.sum() * (1 / period)
    return np.exp(annualized_log_returns) - 1


def get_annualized_log_std(
    log_returns: pd.DataFrame, period: Union[str, int, float]
) -> np.ndarray:
    """
    Calculate annualized standard deviation using log returns.

    Args:
        log_returns (pd.DataFrame): DataFrame containing log returns.
        period (Union[str, int, float]): Time period for annualization.

    Returns:
        np.ndarray: Array of annualized standard deviations.

    Raises:
        ValueError: If the period is invalid.
    """
    if isinstance(period, str) and period in PERIOD_MAP:
        period = PERIOD_MAP[period]
    elif not isinstance(period, (int, float)) or period <= 0:
        raise ValueError(f"Invalid period '{period}'.")

    return log_returns.std(ddof=0) * np.sqrt(252 / period)


def log_std_to_pct_std(log_std: float) -> float:
    """
    Convert log standard deviation to percentage standard deviation.

    Args:
        log_std (float): Log standard deviation.

    Returns:
        float: Percentage standard deviation.
    """
    if np.any(log_std == 0):
        return 0.0  # No volatility means no percentage std deviation
    return np.sqrt(np.exp(log_std**2) - 1)


def weighted_average(
    values: Union[pd.Series, np.ndarray], weights: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate weighted average return for the portfolio.

    Args:
        values (Union[pd.Series, np.ndarray]): Values to average.
        weights (Union[pd.Series, np.ndarray]): Weights for averaging.

    Returns:
        float: Weighted average.
    """
    if not isinstance(values, pd.Series):
        return values

    values, weights = np.array(values), np.array(weights)
    weights = weights / sum(weights)
    return np.average(values, weights=weights)


def get_var(profits: np.ndarray, percentile: float) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        profits (np.ndarray): Array of profit/loss values.
        percentile (float): Percentile for VaR calculation.

    Returns:
        float: Value at Risk.
    """
    var = -np.percentile(profits, percentile)
    return var


def split_array_at_value(
    arr: np.ndarray, value: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the array at the closest index to `value`.

    Args:
        arr (np.ndarray): Array to split.
        value (float): Value to split at.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays split at the closest index to `value`.
    """
    index = np.searchsorted(arr, value)
    return arr[:index], arr[index:]


def get_es(pre_var: np.ndarray) -> float:
    """
    Calculate Expected Shortfall (ES).

    Args:
        pre_var (np.ndarray): Array of values below the VaR threshold.

    Returns:
        float: Expected Shortfall.
    """
    return -np.mean(pre_var)

def gbm(
    T: float,
    N: int,
    mu: float,
    sigma: float,
    S0: float | List[float],
    n_sims: int = 1,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Vectorized Geometric Brownian Motion simulation.

    Args:
        T (float): Total time.
        N (int): Number of time steps.
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
        S0 (float | List[float]): Initial stock price.
        n_sims (int): Number of simulations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Time vector, and all simulated paths.
    """
    if type(S0) != float :
        S0 = np.asarray(S0) 

    dt = T / N
    t = np.linspace(0, T, N + 1)  # Ensure time vector aligns with simulated paths

    # Generate Brownian motion increments
    epsilon = np.random.normal(0, 1, (n_sims, N))
    dW = epsilon * np.sqrt(dt)

    # Compute cumulative Brownian motion
    W_t = np.cumsum(dW, axis=1)
    W_t = np.hstack((np.zeros((n_sims, 1)), W_t))  # Ensure W_0 = 0

    # Compute GBM paths
    S_t = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W_t)

    return t, S_t

def sample(values : np.ndarray, start : int, stop : int) -> np.ndarray :
    return [st[start:stop] for st in values]