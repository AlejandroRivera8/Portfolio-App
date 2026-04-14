import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis, norm, probplot
from datetime import date, timedelta

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Interactive Portfolio Analytics App",
    layout="wide"
)

# -----------------------------
# Constants
# -----------------------------
TRADING_DAYS = 252
BENCHMARK = "^GSPC"

# -----------------------------
# Session state setup
# -----------------------------
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if "stored_tickers" not in st.session_state:
    st.session_state.stored_tickers = None

if "stored_start_date" not in st.session_state:
    st.session_state.stored_start_date = None

if "stored_end_date" not in st.session_state:
    st.session_state.stored_end_date = None

if "stored_rf_annual" not in st.session_state:
    st.session_state.stored_rf_annual = None


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(ttl=3600)
def download_price_data(tickers, start_date, end_date):
    all_tickers = list(dict.fromkeys(tickers + [BENCHMARK]))

    try:
        raw = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )
    except Exception as e:
        raise RuntimeError(f"Data download failed: {e}")

    if raw.empty:
        raise RuntimeError("No data was downloaded. Check the ticker symbols and date range.")

    if "Adj Close" in raw.columns:
        prices = raw["Adj Close"].copy()
    else:
        if "Close" in raw.columns:
            prices = raw["Close"].copy()
        else:
            raise RuntimeError("Adjusted close prices were not available from yfinance.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=all_tickers[0])

    prices = prices.sort_index()

    invalid_tickers = []
    for ticker in all_tickers:
        if ticker not in prices.columns:
            invalid_tickers.append(ticker)

    for col in prices.columns:
        if prices[col].dropna().empty:
            invalid_tickers.append(col)

    invalid_tickers = list(set([t for t in invalid_tickers if t != BENCHMARK]))

    if BENCHMARK not in prices.columns or prices[BENCHMARK].dropna().empty:
        raise RuntimeError("Benchmark (^GSPC) failed to download properly.")

    benchmark_prices = prices[[BENCHMARK]].copy()

    valid_user_cols = [
        col for col in prices.columns
        if col != BENCHMARK and col not in invalid_tickers and not prices[col].dropna().empty
    ]

    if len(valid_user_cols) < 3:
        raise RuntimeError(
            f"Need at least 3 valid tickers after validation. Valid tickers found: {valid_user_cols}"
        )

    working = prices[valid_user_cols + [BENCHMARK]].copy()

    user_missing_pct = working[valid_user_cols].isna().mean()
    dropped_tickers = user_missing_pct[user_missing_pct > 0.05].index.tolist()

    if dropped_tickers:
        valid_user_cols = [t for t in valid_user_cols if t not in dropped_tickers]

    if len(valid_user_cols) < 3:
        raise RuntimeError(
            f"After dropping tickers with >5% missing values, fewer than 3 valid tickers remain. "
            f"Remaining tickers: {valid_user_cols}"
        )

    working = working[valid_user_cols + [BENCHMARK]].copy()

    overlap_note = None
    overlap_start = working.apply(lambda col: col.first_valid_index()).max()
    overlap_end = working.apply(lambda col: col.last_valid_index()).min()

    if overlap_start is None or overlap_end is None or overlap_start >= overlap_end:
        raise RuntimeError("Could not find a usable overlapping date range across the selected assets.")

    original_start = pd.to_datetime(start_date)
    original_end = pd.to_datetime(end_date)

    if overlap_start > original_start or overlap_end < original_end:
        overlap_note = (
            f"Data was truncated to the overlapping date range: "
            f"{overlap_start.date()} to {overlap_end.date()}."
        )

    working = working.loc[overlap_start:overlap_end].dropna()

    if working.empty:
        raise RuntimeError("No aligned price data remained after cleaning.")

    return working, benchmark_prices, invalid_tickers, dropped_tickers, overlap_note


@st.cache_data(ttl=3600)
def compute_returns(prices):
    return prices.pct_change().dropna()


@st.cache_data(ttl=3600)
def summary_statistics(returns):
    stats_df = pd.DataFrame(index=returns.columns)
    stats_df["Annualized Mean Return"] = returns.mean() * TRADING_DAYS
    stats_df["Annualized Volatility"] = returns.std() * np.sqrt(TRADING_DAYS)
    stats_df["Skewness"] = returns.apply(skew)
    stats_df["Kurtosis"] = returns.apply(kurtosis)
    stats_df["Minimum Daily Return"] = returns.min()
    stats_df["Maximum Daily Return"] = returns.max()
    return stats_df


@st.cache_data(ttl=3600)
def cumulative_wealth(returns, initial_wealth=10000):
    return initial_wealth * (1 + returns).cumprod()


@st.cache_data(ttl=3600)
def rolling_volatility(returns, window):
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


@st.cache_data(ttl=3600)
def compute_drawdown(return_series):
    wealth_index = (1 + return_series).cumprod()
    running_peak = wealth_index.cummax()
    drawdown = (wealth_index / running_peak) - 1
    max_dd = drawdown.min()
    return drawdown, max_dd


@st.cache_data(ttl=3600)
def risk_adjusted_metrics(returns, rf_annual):
    rf_daily = rf_annual / TRADING_DAYS

    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

    excess_daily = returns.subtract(rf_daily)
    sharpe = (excess_daily.mean() / returns.std()) * np.sqrt(TRADING_DAYS)

    downside = returns.copy()
    downside = downside.subtract(rf_daily)
    downside = downside.where(downside < 0, 0)

    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = (ann_return - rf_annual) / downside_dev.replace(0, np.nan)

    metrics = pd.DataFrame({
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino
    })

    return metrics


def validate_ticker_input(ticker_text):
    raw_tickers = [t.strip().upper() for t in ticker_text.split(",")]
    tickers = [t for t in raw_tickers if t]

    if len(tickers) < 3:
        return None, "Please enter at least 3 ticker symbols."
    if len(tickers) > 10:
        return None, "Please enter no more than 10 ticker symbols."

    unique_tickers = list(dict.fromkeys(tickers))
    if len(unique_tickers) < 3:
        return None, "Please enter at least 3 unique ticker symbols."

    return unique_tickers, None


# -----------------------------
# App title / intro
# -----------------------------
st.title("Interactive Portfolio Analytics Application")
st.markdown(
    """
    Build and analyze stock portfolios with return, risk, and exploratory analytics.
    This version currently includes:
    - User input and data retrieval
    - Return computation and exploratory analysis
    - Risk analysis
    """
)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Portfolio Inputs")

default_start = date.today() - timedelta(days=365 * 5)
default_end = date.today()

ticker_text = st.sidebar.text_input(
    "Enter 3 to 10 stock tickers (comma-separated)",
    value="AAPL, MSFT, NVDA, AMZN"
)

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=default_end)

rf_percent = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1
)
rf_annual = rf_percent / 100.0

# -----------------------------
# Run button logic
# -----------------------------
if st.sidebar.button("Run Analysis", type="primary"):
    tickers, error_msg = validate_ticker_input(ticker_text)

    if error_msg:
        st.error(error_msg)
        st.session_state.analysis_ready = False
        st.stop()

    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.session_state.analysis_ready = False
        st.stop()

    if (end_date - start_date).days < 730:
        st.error("Please select a date range of at least 2 years.")
        st.session_state.analysis_ready = False
        st.stop()

    st.session_state.analysis_ready = True
    st.session_state.stored_tickers = tickers
    st.session_state.stored_start_date = start_date
    st.session_state.stored_end_date = end_date
    st.session_state.stored_rf_annual = rf_annual

# -----------------------------
# Main app logic
# -----------------------------
if st.session_state.analysis_ready:
    tickers = st.session_state.stored_tickers
    start_date = st.session_state.stored_start_date
    end_date = st.session_state.stored_end_date
    rf_annual = st.session_state.stored_rf_annual

    with st.spinner("Downloading and processing market data..."):
        try:
            prices, benchmark_prices, invalid_tickers, dropped_tickers, overlap_note = download_price_data(
                tickers, start_date, end_date
            )
            returns = compute_returns(prices)
            stats_df = summary_statistics(returns)
            wealth = cumulative_wealth(returns)
            metrics_df = risk_adjusted_metrics(returns, rf_annual)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if invalid_tickers:
        st.warning(f"These tickers failed to download or had insufficient data: {', '.join(invalid_tickers)}")

    if dropped_tickers:
        st.warning(f"These tickers were dropped because they had more than 5% missing values: {', '.join(dropped_tickers)}")

    if overlap_note:
        st.info(overlap_note)

    user_assets = [col for col in prices.columns if col != BENCHMARK]

    tab1, tab2, tab3 = st.tabs([
        "Data Overview",
        "Return Analysis",
        "Risk Analysis"
    ])

    with tab1:
        st.subheader("Cleaned Price Data")
        st.dataframe(prices.tail(), use_container_width=True)

        st.subheader("Daily Return Data")
        st.dataframe(returns.tail(), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Assets", len(user_assets))
        col2.metric("Observations", len(prices))
        col3.metric("Benchmark", BENCHMARK)

    with tab2:
        st.subheader("Summary Statistics")
        st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Cumulative Wealth Index")
        selected_series = st.multiselect(
            "Select assets to display",
            options=list(wealth.columns),
            default=list(wealth.columns)
        )

        if selected_series:
            wealth_plot_df = wealth[selected_series].copy()
            fig_wealth = px.line(
                wealth_plot_df,
                x=wealth_plot_df.index,
                y=wealth_plot_df.columns,
                title="Growth of $10,000",
                labels={"value": "Portfolio Value ($)", "index": "Date", "variable": "Series"}
            )
            fig_wealth.update_layout(legend_title_text="Assets")
            st.plotly_chart(fig_wealth, use_container_width=True)
        else:
            st.warning("Please select at least one asset for the wealth chart.")

        st.subheader("Distribution Analysis")

        selected_stock = st.selectbox(
            "Select a stock for distribution analysis",
            options=user_assets
        )

        plot_type = st.radio(
            "Choose view",
            options=["Histogram with Normal Curve", "Q-Q Plot"],
            horizontal=True
        )

        stock_returns = returns[selected_stock].dropna()

        if plot_type == "Histogram with Normal Curve":
            hist_fig = go.Figure()

            hist_fig.add_trace(go.Histogram(
                x=stock_returns,
                histnorm="probability density",
                name="Daily Returns",
                nbinsx=50
            ))

            x_vals = np.linspace(stock_returns.min(), stock_returns.max(), 500)
            y_vals = norm.pdf(x_vals, stock_returns.mean(), stock_returns.std())

            hist_fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Fitted Normal Curve"
            ))

            hist_fig.update_layout(
                title=f"Return Distribution: {selected_stock}",
                xaxis_title="Daily Return",
                yaxis_title="Density"
            )
            st.plotly_chart(hist_fig, use_container_width=True)

        else:
            qq = probplot(stock_returns, dist="norm")
            theoretical_quantiles = qq[0][0]
            ordered_returns = qq[0][1]

            qq_fig = go.Figure()

            qq_fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=ordered_returns,
                mode="markers",
                name="Observed Quantiles"
            ))

            slope, intercept = qq[1][0], qq[1][1]
            line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
            line_y = slope * line_x + intercept

            qq_fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Reference Line"
            ))

            qq_fig.update_layout(
                title=f"Q-Q Plot: {selected_stock}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Observed Quantiles"
            )
            st.plotly_chart(qq_fig, use_container_width=True)

    with tab3:
        st.subheader("Rolling Annualized Volatility")

        vol_window = st.select_slider(
            "Select rolling window length",
            options=[30, 60, 90, 120],
            value=60
        )

        rolling_vol = rolling_volatility(returns[user_assets], vol_window)

        fig_roll_vol = px.line(
            rolling_vol,
            x=rolling_vol.index,
            y=rolling_vol.columns,
            title=f"Rolling Annualized Volatility ({vol_window}-Day Window)",
            labels={"value": "Annualized Volatility", "index": "Date", "variable": "Ticker"}
        )
        st.plotly_chart(fig_roll_vol, use_container_width=True)

        st.subheader("Drawdown Analysis")

        dd_stock = st.selectbox(
            "Select a stock for drawdown analysis",
            options=user_assets,
            key="drawdown_stock"
        )

        dd_series, max_dd = compute_drawdown(returns[dd_stock])

        dd_col1, dd_col2 = st.columns([1, 3])
        dd_col1.metric("Maximum Drawdown", f"{max_dd:.2%}")

        fig_dd = px.line(
            x=dd_series.index,
            y=dd_series.values,
            title=f"Drawdown Chart: {dd_stock}",
            labels={"x": "Date", "y": "Drawdown"}
        )
        fig_dd.update_yaxes(tickformat=".1%")
        dd_col2.plotly_chart(fig_dd, use_container_width=True)

        st.subheader("Risk-Adjusted Metrics")
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

else:
    st.info("Enter your portfolio inputs in the sidebar and click Run Analysis.")