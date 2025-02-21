# Portfolio Risk Estimation

## Overview

This project is a **Streamlit** application designed to estimate portfolio risk using **Monte Carlo simulation**. It allows users to input their asset holdings, select historical periods for analysis, and apply market shocks to stress-test portfolio performance. The app computes key risk metrics such as **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** and provides visual insights through simulations and histograms.

## Features

-   **Dynamic Ticker Selection**: Users can search and select multiple assets from most available large Yahoo Finance tickers.
-   **Market Shock Analysis**: Users can introduce custom shocks to assess portfolio resilience.
-   **Monte Carlo Simulation**: Simulates thousands of future portfolio paths based on historical log returns and volatility.
-   **Risk Metrics**: Computes **VaR** and **CVaR** for different time horizons.
-   **Visual Analytics**: Generates simulation plots and histograms of projected portfolio values and losses.

## Installation

### Prerequisites

-   Python 3.8+
-   Streamlit
-   NumPy
-   Pandas
-   Matplotlib
-   YFinance
-   Requests
-   Python-dotenv
-   Python-dateutil
-   Matplotlib

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/portfolio-risk-estimation.git
    cd portfolio-risk-estimation
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Enter Portfolio Details**: Select tickers and input asset values.
2. **Choose Historical Data Period**: Determines returns and volatility calculations.
3. **Apply Market Shocks**: Adjust shocks' magnitude and duation for stress testing.
4. **Select Time Horizon**: Define sample periods for **VaR** and **CVaR** computations.
5. **Run Monte Carlo Simulation**: View risk analytics and performance distributions.

## Future Improvements

-   Implement **automatic assessment of portfolio weaknesses**.
-   Enhance market shock analysis by comparing affected vs. baseline portfolio performance.
-   Introduce **correlation-based risk clustering**.

## Contributing

Contributions are welcome! Feel free to submit pull requests or report issues.
