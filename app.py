import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
from portfolio import Portfolio
from math import isnan


# TODO! IMPLEMENT MARKET SHOCKS, ANALYSE VS BASELINE, ANALYSE WHICH PART OF PORTFOLIO IS WEAK
#! MAYBE AUTOMATIC ASSESSMENT OF WEAKNESSES

YEARLY_TRADING_DAYS = 252
TIME_HORIZON = 1
N_SIMS = 50_000
RENDERED_SIMS = 500
VAR_PERCENTILE = 5
FEAR_FACTOR = 0.5
IR_INDEX = 1


def main():
    TICKERS_FILE_PATH = "data/tickers.csv"
    SHOCKS_FILE_PATH = "data/shocks.csv"
    TICKERS = pd.read_csv(TICKERS_FILE_PATH)
    SHOCKS = pd.read_csv(SHOCKS_FILE_PATH)

    WIFI = check_wifi_connection()
    API = check_api_key()

    if not API:
        SHOCKS = SHOCKS.drop(IR_INDEX)

    TICKER_SYMBOLS, TICKER_NAMES = TICKERS["Symbol"], TICKERS["Contract Name"]
    SHOCK_SYMBOLS, SHOCK_NAMES = SHOCKS["Symbol"], SHOCKS["Contract Name"]

    def shock_formatter(name) -> str:
        symbol = SHOCK_SYMBOLS[SHOCK_NAMES == name].values[0]

        if type(symbol) == float and isnan(symbol):
            return name
        else :
            return f"{symbol} - {name}"

    # ---------------- Streamlit UI ---------------- #
    st.title("Portfolio Risk Estimation")

    # Check if user has an active wifi connection, else give a warning and stop the program until reload
    if not WIFI:
        st.warning(
            "No WiFi connection detected. Please check your internet connection and try again."
        )
        return

    # --- Portfolio Input --- #
    st.sidebar.header("Portfolio Details")

    # Searchable Ticker Selection
    selected_tickers = st.sidebar.multiselect(
        "Select Asset Tickers:",
        options=TICKER_SYMBOLS,
        format_func=lambda x: f"{x} - {TICKER_NAMES[TICKER_SYMBOLS == x].values[0]}",
        help="Search and select asset tickers",
    )

    # Input for corresponding asset values (dynamic)
    ticker_values = {}
    for ticker in selected_tickers:
        ticker_values[ticker] = st.sidebar.number_input(
            f"Enter value for {ticker}:",
            min_value=0.0,
            value=1000.0,  # Default value
            step=100.0,
        )

    period = st.sidebar.selectbox(
        "Select historical period:", list(PERIOD_MAP.keys()), index=2
    )

    # Convert user inputs
    values = list(ticker_values.values())


    # Searchable Shock Selection
    selected_shocks = st.sidebar.multiselect(
        "Select Market Shocks:",
        options=SHOCK_NAMES,
        format_func=shock_formatter,
        help="Search and select market shocks",
    )

    shock_duration = None
    return_changes = []
    volatility_changes = []

    if selected_shocks:
        shock_duration = st.sidebar.number_input(
            "Duration (days)",
            value=21,
            step=1,
            min_value=0,
            max_value=YEARLY_TRADING_DAYS,
        )

    # Input for corresponding shock values (dynamic)
    shock_values = {}
    for shock in selected_shocks:
        st.sidebar.write(f"{SHOCK_NAMES[SHOCK_SYMBOLS == shock].values[0]}:")
        # Input for corresponding market shock (dynamic)
        col1, _ = st.sidebar.columns(2)
        # Add number inputs to each column
        with col1:
            magnitude = st.number_input("Magnitude (%)", value=0.1, step=0.01)
            magnitude /= 100
            shock_values[shock] = magnitude

    # Sidebar Header
    st.sidebar.subheader("Select Sample Time Horizon for metrics (VaR, CVaR)")

    # Start and End Date Pickers
    sample_start = st.sidebar.number_input("Start Date", value=0, min_value=0)
    sample_stop = st.sidebar.number_input("End Date", value=21, min_value=0)
    sample_start_days = sample_start
    sample_stop_days = sample_stop

    # Time Unit Selection
    time_unit = st.sidebar.radio(
        "Select Time Unit:", ["Trading Days", "Weeks", "Months"], index=0
    )

    if time_unit == "Weeks":
        sample_start_days = 5 * sample_start
        sample_stop_days = 5 * sample_stop
    elif time_unit == "Months":
        sample_start_days = 21 * sample_start
        sample_stop_days = 21 * sample_stop

    if sample_stop_days > YEARLY_TRADING_DAYS or sample_start_days > sample_stop_days:
        st.sidebar.warning(
            "Ensure the sample time horizon is valid (exceeding yearly trading days or invalid period)"
        )
    elif selected_tickers and values:

        portfolio = Portfolio(selected_tickers, values, period)

        # Fetch and process historical data
        portfolio.get_historical_data()
        portfolio.get_portfolio_attributes()

        portfolio_return, portfolio_std = (
            portfolio.portfolio_return,
            portfolio.portfolio_std,
        )

        for shock in shock_values:
            magnitude = shock_values[shock]

            if shock == "EM":
                anualized_shock_return = annualize_return(magnitude, shock_duration)
                annualized_shockstd = annualize_std(FEAR_FACTOR * abs(magnitude), shock_duration)

                part = portfolio.get_equity_part(TICKERS)

                return_changes.append(anualized_shock_return * part)
                volatility_changes.append(annualized_shockstd * part)
                

        if shock_values:
            # --- Monte Carlo Simulation --- #
            shock_t, shock_future_values = gbm(
                shock_duration / YEARLY_TRADING_DAYS,
                shock_duration,
                portfolio_return + sum(return_changes),
                portfolio_std + sum(volatility_changes),
                portfolio.initial_portfolio_value,
                N_SIMS,
            )

            last_shock_values = [s[-1] for s in shock_future_values]

            t, future_values = gbm(
                TIME_HORIZON - shock_duration / YEARLY_TRADING_DAYS,
                YEARLY_TRADING_DAYS - shock_duration,
                portfolio_return,
                portfolio_std,
                last_shock_values,
                N_SIMS,
            )
            future_values = np.concat((shock_future_values, future_values), axis=1)
            t += shock_duration / YEARLY_TRADING_DAYS
            t = np.concat((shock_t, t))
        else:
            t, future_values = gbm(
                TIME_HORIZON,
                YEARLY_TRADING_DAYS,
                portfolio_return,
                portfolio_std,
                portfolio.initial_portfolio_value,
                N_SIMS,
            )

        sample = sample_values(future_values, sample_start_days, sample_stop_days)

        profits_sample = np.asarray(sorted([i[-1] - i[0] for i in sample]))
        final_values_total = np.asarray([i[-1] for i in future_values])

        var_sample = int(get_var(profits_sample, VAR_PERCENTILE))
        pre_var_sample, post_var_sample = split_array_at_value(
            profits_sample, -var_sample
        )

        es = int(get_es(pre_var_sample))

        # --- Sidebar Portfolio Statistics --- #
        st.sidebar.subheader("Portfolio Statistics")

        st.sidebar.write(
            f"Initial portfolio value: **${int(portfolio.initial_portfolio_value):,}**"
        )
        st.sidebar.write(f"Predicted Annual Return: **{portfolio_return:.2%}**")
        st.sidebar.write(f"Predicted Volatility: **{portfolio_std:.2%}**")

        # --------------------------------------------------

        # --- Monte Carlo Simulation Plot --- #
        st.subheader("Monte Carlo Simulation")
        fig, _ = plt.subplots(figsize=(10, 5))
        plt.plot(
            t, np.array(future_values[:RENDERED_SIMS]).T, alpha=0.3
        )  # Plot fewer lines
        plt.axhline(
            y=portfolio.initial_portfolio_value,
            color="purple",
            linestyle="dashed",
            label="Initial Value",
        )
        plt.title("Portfolio Value Simulation")
        plt.xlabel("Time (Years)")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        st.pyplot(fig)

        col1, col2 = st.columns(2)

        # --- Final Portfolio Values Histogram --- #
        with col1:
            st.subheader("Final Portfolio Values")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.hist(
                final_values_total, bins=30, color="black", alpha=0.7, edgecolor="black"
            )
            ax1.axvline(
                x=portfolio.initial_portfolio_value,
                color="purple",
                linestyle="dashed",
                label="Initial Value",
            )
            ax1.set_xlabel("Final Portfolio Value ($)")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Histogram of Final Portfolio Values")
            ax1.legend()
            st.pyplot(fig1)

        # --- VaR and CVaR Histogram --- #
        with col2:
            st.subheader(
                f"VaR and CVaR Analysis ({time_unit} {sample_start}-{sample_stop})"
            )
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.hist(
                pre_var_sample,
                bins=12,
                color="red",
                alpha=0.7,
                edgecolor="black",
                label="Under VaR",
            )
            ax2.hist(
                post_var_sample,
                bins=50,
                color="blue",
                alpha=0.7,
                edgecolor="black",
                label="Over VaR",
            )
            ax2.axvline(
                x=-es,
                color="black",
                linestyle="dashed",
                label="Expected Shortfall (CVaR)",
            )
            ax2.set_xlabel("Profit ($)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Histogram of Profits")
            ax2.legend()
            st.pyplot(fig2)

        st.subheader(f"Analytics over {time_unit} {sample_start}-{sample_stop}")
        st.markdown(f"##### {VAR_PERCENTILE}% Value at Risk (VaR): ${var_sample:,}")
        st.markdown(f"##### {VAR_PERCENTILE}% Expected Shortfall (CVaR): ${es:,}")

    else:
        st.info("Enter portfolio details in the sidebar to begin.")


if __name__ == "__main__":
    main()
