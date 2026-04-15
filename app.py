import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis, norm, probplot
from scipy.optimize import minimize
from datetime import date, timedelta

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Interactive Portfolio Analytics Application",
    layout="wide"
)

# =========================================================
# CONSTANTS
# =========================================================
TRADING_DAYS = 252
BENCHMARK = "^GSPC"

# =========================================================
# SESSION STATE
# =========================================================
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

# =========================================================
# HELPER FUNCTIONS
# =========================================================
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


@st.cache_data(ttl=3600)
def download_price_data(tickers, start_date, end_date):
    all_tickers = list(dict.fromkeys(tickers + [BENCHMARK]))

    try:
        raw = yf.download(
            tickers=all_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=True
        )
    except Exception as e:
        raise RuntimeError(f"Data download failed: {e}")

    if raw is None or raw.empty:
        raise RuntimeError("No data was downloaded. Check your tickers and date range.")

    prices = None

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = [str(x) for x in raw.columns.get_level_values(0)]
        if "Adj Close" in level0:
            prices = raw["Adj Close"].copy()
        elif "Close" in level0:
            prices = raw["Close"].copy()
    else:
        if "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy()
            if len(all_tickers) == 1:
                prices.columns = [all_tickers[0]]
        elif "Close" in raw.columns:
            prices = raw[["Close"]].copy()
            if len(all_tickers) == 1:
                prices.columns = [all_tickers[0]]

    if prices is None:
        raise RuntimeError("Could not find adjusted close or close prices in the yfinance output.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.sort_index()
    prices.columns = [str(col) for col in prices.columns]

    invalid_tickers = []
    for ticker in all_tickers:
        if ticker not in prices.columns:
            if ticker != BENCHMARK:
                invalid_tickers.append(ticker)
        elif prices[ticker].dropna().empty:
            if ticker != BENCHMARK:
                invalid_tickers.append(ticker)

    if BENCHMARK not in prices.columns or prices[BENCHMARK].dropna().empty:
        raise RuntimeError("Benchmark (^GSPC) failed to download properly.")

    valid_user_cols = [
        col for col in prices.columns
        if col != BENCHMARK and col not in invalid_tickers and not prices[col].dropna().empty
    ]

    if len(valid_user_cols) < 3:
        raise RuntimeError(
            f"Need at least 3 valid tickers after validation. Valid tickers found: {valid_user_cols}"
        )

    working = prices[valid_user_cols + [BENCHMARK]].copy()

    missing_pct = working[valid_user_cols].isna().mean()
    dropped_tickers = missing_pct[missing_pct > 0.05].index.tolist()

    if dropped_tickers:
        valid_user_cols = [t for t in valid_user_cols if t not in dropped_tickers]

    if len(valid_user_cols) < 3:
        raise RuntimeError(
            f"After dropping tickers with >5% missing values, fewer than 3 valid tickers remain. "
            f"Remaining tickers: {valid_user_cols}"
        )

    working = prices[valid_user_cols + [BENCHMARK]].copy()

    overlap_start = working.apply(lambda col: col.first_valid_index()).max()
    overlap_end = working.apply(lambda col: col.last_valid_index()).min()

    if overlap_start is None or overlap_end is None or overlap_start >= overlap_end:
        raise RuntimeError("Could not find a usable overlapping date range across the selected assets.")

    original_start = pd.to_datetime(start_date)
    original_end = pd.to_datetime(end_date)

    overlap_note = None
    if overlap_start > original_start or overlap_end < original_end:
        overlap_note = (
            f"Data was truncated to the overlapping date range: "
            f"{overlap_start.date()} to {overlap_end.date()}."
        )

    working = working.loc[overlap_start:overlap_end].dropna()

    if working.empty:
        raise RuntimeError("No aligned price data remained after cleaning.")

    return working, invalid_tickers, dropped_tickers, overlap_note


@st.cache_data(ttl=3600)
def compute_returns(prices):
    return prices.pct_change().dropna()


def summary_statistics(returns):
    stats_df = pd.DataFrame(index=returns.columns)
    stats_df["Annualized Mean Return"] = returns.mean() * TRADING_DAYS
    stats_df["Annualized Volatility"] = returns.std() * np.sqrt(TRADING_DAYS)
    stats_df["Skewness"] = returns.apply(skew)
    stats_df["Kurtosis"] = returns.apply(kurtosis)
    stats_df["Minimum Daily Return"] = returns.min()
    stats_df["Maximum Daily Return"] = returns.max()
    return stats_df


def cumulative_wealth(returns, initial_wealth=10000):
    return initial_wealth * (1 + returns).cumprod()


def rolling_volatility(returns, window):
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def compute_drawdown(return_series):
    wealth_index = (1 + return_series).cumprod()
    running_peak = wealth_index.cummax()
    drawdown = (wealth_index / running_peak) - 1
    max_dd = drawdown.min()
    return drawdown, max_dd


def risk_adjusted_metrics(returns, rf_annual):
    rf_daily = rf_annual / TRADING_DAYS

    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

    sharpe = ((returns - rf_daily).mean() / returns.std()) * np.sqrt(TRADING_DAYS)

    downside_excess = returns - rf_daily
    downside_excess = downside_excess.where(downside_excess < 0, 0)

    downside_dev = np.sqrt((downside_excess ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = (ann_return - rf_annual) / downside_dev.replace(0, np.nan)

    metrics = pd.DataFrame({
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino
    })

    return metrics


def portfolio_annual_return(weights, mean_daily_returns):
    return float(np.sum(weights * mean_daily_returns) * TRADING_DAYS)


def portfolio_annual_volatility(weights, cov_matrix_daily):
    cov_annual = cov_matrix_daily * TRADING_DAYS
    return float(np.sqrt(weights.T @ cov_annual @ weights))


def portfolio_daily_returns(asset_returns, weights):
    return asset_returns @ weights


def portfolio_sortino_ratio(asset_returns, weights, rf_annual):
    rf_daily = rf_annual / TRADING_DAYS
    p_returns = portfolio_daily_returns(asset_returns, weights)

    ann_return = p_returns.mean() * TRADING_DAYS
    downside_excess = p_returns - rf_daily
    downside_excess = np.where(downside_excess < 0, downside_excess, 0)

    downside_dev = np.sqrt(np.mean(downside_excess ** 2)) * np.sqrt(TRADING_DAYS)

    if downside_dev == 0:
        return np.nan

    return float((ann_return - rf_annual) / downside_dev)


def portfolio_max_drawdown(asset_returns, weights):
    p_returns = portfolio_daily_returns(asset_returns, weights)
    _, max_dd = compute_drawdown(p_returns)
    return float(max_dd)


def portfolio_sharpe_ratio(weights, mean_daily_returns, cov_matrix_daily, rf_annual):
    port_return = portfolio_annual_return(weights, mean_daily_returns)
    port_vol = portfolio_annual_volatility(weights, cov_matrix_daily)

    if port_vol == 0:
        return np.nan

    return float((port_return - rf_annual) / port_vol)


def negative_sharpe_ratio(weights, mean_daily_returns, cov_matrix_daily, rf_annual):
    sharpe = portfolio_sharpe_ratio(weights, mean_daily_returns, cov_matrix_daily, rf_annual)
    if np.isnan(sharpe):
        return 1e6
    return -sharpe


def portfolio_volatility_objective(weights, cov_matrix_daily):
    return portfolio_annual_volatility(weights, cov_matrix_daily)


def optimize_gmv(mean_daily_returns, cov_matrix_daily, asset_names):
    n_assets = len(asset_names)
    initial_weights = np.array([1 / n_assets] * n_assets)

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        portfolio_volatility_objective,
        initial_weights,
        args=(cov_matrix_daily,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result


def optimize_tangency(mean_daily_returns, cov_matrix_daily, asset_names, rf_annual):
    n_assets = len(asset_names)
    initial_weights = np.array([1 / n_assets] * n_assets)

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_daily_returns, cov_matrix_daily, rf_annual),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result


def build_portfolio_summary(asset_returns, mean_daily_returns, cov_matrix_daily, weights, rf_annual, label):
    ann_return = portfolio_annual_return(weights, mean_daily_returns)
    ann_vol = portfolio_annual_volatility(weights, cov_matrix_daily)
    sharpe = portfolio_sharpe_ratio(weights, mean_daily_returns, cov_matrix_daily, rf_annual)
    sortino = portfolio_sortino_ratio(asset_returns, weights, rf_annual)
    max_dd = portfolio_max_drawdown(asset_returns, weights)

    return pd.Series({
        "Portfolio": label,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": max_dd
    })


def make_weights_df(asset_names, eq_weights, gmv_weights, tan_weights, custom_weights):
    return pd.DataFrame({
        "Asset": asset_names,
        "Equal Weight": eq_weights,
        "GMV Weight": gmv_weights,
        "Tangency Weight": tan_weights,
        "Custom Weight": custom_weights
    })


def risk_contribution(weights, cov_matrix_daily):
    cov_annual = cov_matrix_daily * TRADING_DAYS
    port_var = weights.T @ cov_annual @ weights
    if port_var == 0:
        return np.zeros(len(weights))
    rc = (weights * (cov_annual @ weights)) / port_var
    return rc


def efficient_frontier_points(mean_daily_returns, cov_matrix_daily, n_points=40):
    n_assets = len(mean_daily_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    initial_weights = np.array([1 / n_assets] * n_assets)

    asset_ann_returns = mean_daily_returns * TRADING_DAYS
    target_returns = np.linspace(asset_ann_returns.min(), asset_ann_returns.max(), n_points)

    frontier = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_annual_return(w, mean_daily_returns) - t}
        ]

        result = minimize(
            portfolio_volatility_objective,
            initial_weights,
            args=(cov_matrix_daily,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            w = result.x
            frontier.append({
                "Return": portfolio_annual_return(w, mean_daily_returns),
                "Volatility": portfolio_annual_volatility(w, cov_matrix_daily)
            })

    return pd.DataFrame(frontier)


def get_lookback_options(returns_index):
    total_days = (returns_index.max() - returns_index.min()).days
    options = {"Full Sample": None}

    if total_days >= 365:
        options["1 Year"] = 252
    if total_days >= 3 * 365:
        options["3 Years"] = 252 * 3
    if total_days >= 5 * 365:
        options["5 Years"] = 252 * 5

    ordered = {}
    for key in ["1 Year", "3 Years", "5 Years", "Full Sample"]:
        if key in options:
            ordered[key] = options[key]
    return ordered


def run_sensitivity_analysis(asset_returns_only, asset_names, rf_annual, lookback_options):
    results = []
    weights_rows = []

    for label, lookback_days in lookback_options.items():
        if lookback_days is None:
            subset = asset_returns_only.copy()
        else:
            subset = asset_returns_only.tail(lookback_days).copy()

        if len(subset) < 60:
            continue

        mean_daily_returns = subset.mean()
        cov_matrix = subset.cov()

        gmv_result = optimize_gmv(mean_daily_returns, cov_matrix, tuple(asset_names))
        tangency_result = optimize_tangency(mean_daily_returns, cov_matrix, tuple(asset_names), rf_annual)

        if not gmv_result.success or not tangency_result.success:
            continue

        gmv_weights = gmv_result.x
        tangency_weights = tangency_result.x

        gmv_return = portfolio_annual_return(gmv_weights, mean_daily_returns)
        gmv_vol = portfolio_annual_volatility(gmv_weights, cov_matrix)

        tangency_return = portfolio_annual_return(tangency_weights, mean_daily_returns)
        tangency_vol = portfolio_annual_volatility(tangency_weights, cov_matrix)
        tangency_sharpe = portfolio_sharpe_ratio(tangency_weights, mean_daily_returns, cov_matrix, rf_annual)

        results.append({
            "Window": label,
            "GMV Annualized Return": gmv_return,
            "GMV Annualized Volatility": gmv_vol,
            "Tangency Annualized Return": tangency_return,
            "Tangency Annualized Volatility": tangency_vol,
            "Tangency Sharpe Ratio": tangency_sharpe
        })

        for asset, w in zip(asset_names, gmv_weights):
            weights_rows.append({
                "Window": label,
                "Portfolio": "GMV",
                "Asset": asset,
                "Weight": w
            })

        for asset, w in zip(asset_names, tangency_weights):
            weights_rows.append({
                "Window": label,
                "Portfolio": "Tangency",
                "Asset": asset,
                "Weight": w
            })

    results_df = pd.DataFrame(results)
    weights_df = pd.DataFrame(weights_rows)

    return results_df, weights_df


# =========================================================
# TITLE / INTRO
# =========================================================
st.title("Interactive Portfolio Analytics Application")
st.markdown(
    """
Build and analyze stock portfolios with return, risk, and exploratory analytics.

This version currently includes:
- User input and data retrieval
- Return computation and exploratory analysis
- Risk analysis
- Correlation analysis
- Portfolio optimization
- Estimation window sensitivity
"""
)

# =========================================================
# SIDEBAR INPUTS
# =========================================================
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

with st.sidebar.expander("About / Methodology"):
    st.markdown(
        """
- **Data source:** yfinance adjusted close prices
- **Return type:** simple daily returns
- **Annualized return:** mean daily return × 252
- **Annualized volatility:** daily standard deviation × √252
- **Risk-free rate:** annual input divided by 252 where needed
- **Sharpe ratio:** excess return over total volatility
- **Sortino ratio:** excess return over downside deviation
- **Cumulative wealth:** $10,000 × (1 + returns).cumprod()
- **Optimization constraints:** no short selling, weights between 0 and 1, weights sum to 1
"""
    )

# =========================================================
# RUN BUTTON LOGIC
# =========================================================
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

# =========================================================
# MAIN APP
# =========================================================
if st.session_state.analysis_ready:
    tickers = st.session_state.stored_tickers
    start_date = st.session_state.stored_start_date
    end_date = st.session_state.stored_end_date
    rf_annual = st.session_state.stored_rf_annual

    with st.spinner("Downloading and processing market data..."):
        try:
            prices, invalid_tickers, dropped_tickers, overlap_note = download_price_data(
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
        st.warning(
            f"These tickers failed to download or had insufficient data: {', '.join(invalid_tickers)}"
        )

    if dropped_tickers:
        st.warning(
            f"These tickers were dropped because they had more than 5% missing values: {', '.join(dropped_tickers)}"
        )

    if overlap_note:
        st.info(overlap_note)

    user_assets = [col for col in prices.columns if col != BENCHMARK]
    asset_returns_only = returns[user_assets]
    mean_daily_returns = asset_returns_only.mean()
    cov_matrix = asset_returns_only.cov()
    corr_matrix = asset_returns_only.corr()

    n_assets = len(user_assets)
    eq_weights = np.array([1 / n_assets] * n_assets)

    # -------------------------
    # Custom portfolio weights
    # -------------------------
    raw_custom_weights = []
    with st.sidebar.expander("Custom Portfolio Builder", expanded=False):
        st.markdown("Set custom raw weights below. They will be normalized to sum to 100%.")
        for ticker in user_assets:
            default_slider_value = int(round(100 / n_assets))
            raw_w = st.slider(
                f"{ticker} raw weight",
                min_value=0,
                max_value=100,
                value=default_slider_value,
                step=1,
                key=f"custom_{ticker}"
            )
            raw_custom_weights.append(raw_w)

    raw_custom_weights = np.array(raw_custom_weights, dtype=float)
    if raw_custom_weights.sum() == 0:
        custom_weights = eq_weights.copy()
    else:
        custom_weights = raw_custom_weights / raw_custom_weights.sum()

    # -------------------------
    # Core portfolios
    # -------------------------
    eq_summary = build_portfolio_summary(
        asset_returns_only,
        mean_daily_returns,
        cov_matrix,
        eq_weights,
        rf_annual,
        "Equal Weight"
    )

    gmv_result = optimize_gmv(mean_daily_returns, cov_matrix, tuple(user_assets))
    tangency_result = optimize_tangency(mean_daily_returns, cov_matrix, tuple(user_assets), rf_annual)

    if not gmv_result.success:
        st.error("Global Minimum Variance optimization failed.")
        st.stop()

    if not tangency_result.success:
        st.error("Tangency portfolio optimization failed.")
        st.stop()

    gmv_weights = gmv_result.x
    tangency_weights = tangency_result.x

    gmv_summary = build_portfolio_summary(
        asset_returns_only,
        mean_daily_returns,
        cov_matrix,
        gmv_weights,
        rf_annual,
        "GMV"
    )

    tangency_summary = build_portfolio_summary(
        asset_returns_only,
        mean_daily_returns,
        cov_matrix,
        tangency_weights,
        rf_annual,
        "Tangency"
    )

    custom_summary = build_portfolio_summary(
        asset_returns_only,
        mean_daily_returns,
        cov_matrix,
        custom_weights,
        rf_annual,
        "Custom"
    )

    benchmark_max_dd = compute_drawdown(returns[BENCHMARK])[1]
    benchmark_summary = pd.Series({
        "Portfolio": "S&P 500",
        "Annualized Return": returns[BENCHMARK].mean() * TRADING_DAYS,
        "Annualized Volatility": returns[BENCHMARK].std() * np.sqrt(TRADING_DAYS),
        "Sharpe Ratio": metrics_df.loc[BENCHMARK, "Sharpe Ratio"],
        "Sortino Ratio": metrics_df.loc[BENCHMARK, "Sortino Ratio"],
        "Maximum Drawdown": benchmark_max_dd
    })

    portfolio_summary_df = pd.DataFrame(
        [eq_summary, gmv_summary, tangency_summary, custom_summary, benchmark_summary]
    ).set_index("Portfolio")

    weights_df = make_weights_df(user_assets, eq_weights, gmv_weights, tangency_weights, custom_weights)

    # -------------------------
    # Risk contribution
    # -------------------------
    gmv_rc = risk_contribution(gmv_weights, cov_matrix)
    tangency_rc = risk_contribution(tangency_weights, cov_matrix)

    gmv_rc_df = pd.DataFrame({
        "Asset": user_assets,
        "Weight": gmv_weights,
        "Risk Contribution": gmv_rc
    })

    tangency_rc_df = pd.DataFrame({
        "Asset": user_assets,
        "Weight": tangency_weights,
        "Risk Contribution": tangency_rc
    })

    # -------------------------
    # Efficient frontier
    # -------------------------
    frontier_df = efficient_frontier_points(mean_daily_returns, cov_matrix, n_points=40)

    eq_vol = portfolio_annual_volatility(eq_weights, cov_matrix)
    eq_ret = portfolio_annual_return(eq_weights, mean_daily_returns)

    gmv_vol = portfolio_annual_volatility(gmv_weights, cov_matrix)
    gmv_ret = portfolio_annual_return(gmv_weights, mean_daily_returns)

    tangency_vol = portfolio_annual_volatility(tangency_weights, cov_matrix)
    tangency_ret = portfolio_annual_return(tangency_weights, mean_daily_returns)

    custom_vol = portfolio_annual_volatility(custom_weights, cov_matrix)
    custom_ret = portfolio_annual_return(custom_weights, mean_daily_returns)

    benchmark_vol = returns[BENCHMARK].std() * np.sqrt(TRADING_DAYS)
    benchmark_ret = returns[BENCHMARK].mean() * TRADING_DAYS

    max_frontier_vol = max(
        frontier_df["Volatility"].max() if not frontier_df.empty else 0,
        tangency_vol,
        custom_vol,
        benchmark_vol
    )
    cal_x = np.linspace(0, max_frontier_vol * 1.2, 100)
    tangency_slope = (tangency_ret - rf_annual) / tangency_vol if tangency_vol != 0 else 0
    cal_y = rf_annual + tangency_slope * cal_x

    # -------------------------
    # Portfolio wealth comparison
    # -------------------------
    portfolio_return_series = pd.DataFrame({
        "Equal Weight": portfolio_daily_returns(asset_returns_only, eq_weights),
        "GMV": portfolio_daily_returns(asset_returns_only, gmv_weights),
        "Tangency": portfolio_daily_returns(asset_returns_only, tangency_weights),
        "Custom": portfolio_daily_returns(asset_returns_only, custom_weights),
        "S&P 500": returns[BENCHMARK]
    })

    portfolio_wealth_df = cumulative_wealth(portfolio_return_series, initial_wealth=10000)

    # -------------------------
    # Sensitivity analysis
    # -------------------------
    lookback_options = get_lookback_options(asset_returns_only.index)
    sensitivity_results_df, sensitivity_weights_df = run_sensitivity_analysis(
        asset_returns_only,
        user_assets,
        rf_annual,
        lookback_options
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Data Overview",
        "Return Analysis",
        "Risk Analysis",
        "Correlation Analysis",
        "Portfolio Optimization",
        "Sensitivity Analysis"
    ])

    # =====================================================
    # TAB 1: DATA OVERVIEW
    # =====================================================
    with tab1:
        st.subheader("Cleaned Price Data")
        st.dataframe(prices.tail(), use_container_width=True)

        st.subheader("Daily Return Data")
        st.dataframe(returns.tail(), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Number of Assets", len(user_assets))
        c2.metric("Observations", len(prices))
        c3.metric("Benchmark", BENCHMARK)

    # =====================================================
    # TAB 2: RETURN ANALYSIS
    # =====================================================
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

    # =====================================================
    # TAB 3: RISK ANALYSIS
    # =====================================================
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

    # =====================================================
    # TAB 4: CORRELATION ANALYSIS
    # =====================================================
    with tab4:
        st.subheader("Correlation Heatmap")

        heatmap_fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Pairwise Correlation Matrix of Daily Returns"
        )
        heatmap_fig.update_layout(
            xaxis_title="Assets",
            yaxis_title="Assets",
            coloraxis_colorbar_title="Correlation"
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.subheader("Rolling Correlation")

        col1, col2, col3 = st.columns(3)

        with col1:
            stock_1 = st.selectbox(
                "Select first stock",
                options=user_assets,
                key="rolling_corr_stock_1"
            )

        with col2:
            stock_2_options = [ticker for ticker in user_assets if ticker != stock_1]
            stock_2 = st.selectbox(
                "Select second stock",
                options=stock_2_options,
                key="rolling_corr_stock_2"
            )

        with col3:
            corr_window = st.select_slider(
                "Select rolling window",
                options=[30, 60, 90, 120],
                value=60,
                key="rolling_corr_window"
            )

        rolling_corr_series = returns[stock_1].rolling(corr_window).corr(returns[stock_2])

        rolling_corr_fig = px.line(
            x=rolling_corr_series.index,
            y=rolling_corr_series.values,
            title=f"Rolling Correlation: {stock_1} vs {stock_2} ({corr_window}-Day Window)",
            labels={"x": "Date", "y": "Correlation"}
        )
        rolling_corr_fig.update_yaxes(range=[-1, 1])
        st.plotly_chart(rolling_corr_fig, use_container_width=True)

        with st.expander("Show Covariance Matrix"):
            st.subheader("Covariance Matrix of Daily Returns")
            st.dataframe(cov_matrix.style.format("{:.6f}"), use_container_width=True)

    # =====================================================
    # TAB 5: PORTFOLIO OPTIMIZATION
    # =====================================================
    with tab5:
        st.subheader("Portfolio Summary")

        st.markdown(
            """
This section compares:
- **Equal Weight**
- **GMV (Global Minimum Variance)**
- **Tangency (Maximum Sharpe Ratio)**
- **Custom Portfolio**
- **S&P 500 benchmark**
"""
        )

        st.dataframe(
            portfolio_summary_df.style.format({
                "Annualized Return": "{:.4f}",
                "Annualized Volatility": "{:.4f}",
                "Sharpe Ratio": "{:.4f}",
                "Sortino Ratio": "{:.4f}",
                "Maximum Drawdown": "{:.2%}"
            }),
            use_container_width=True
        )

        st.subheader("Normalized Custom Portfolio Weights")
        custom_weights_display = pd.DataFrame({
            "Asset": user_assets,
            "Normalized Weight": custom_weights
        })
        st.dataframe(
            custom_weights_display.style.format({"Normalized Weight": "{:.2%}"}),
            use_container_width=True
        )

        st.subheader("Portfolio Weights Table")
        st.dataframe(
            weights_df.style.format({
                "Equal Weight": "{:.2%}",
                "GMV Weight": "{:.2%}",
                "Tangency Weight": "{:.2%}",
                "Custom Weight": "{:.2%}"
            }),
            use_container_width=True
        )

        st.subheader("Portfolio Weights Chart")
        weights_plot_df = weights_df.melt(
            id_vars="Asset",
            value_vars=["Equal Weight", "GMV Weight", "Tangency Weight", "Custom Weight"],
            var_name="Portfolio",
            value_name="Weight"
        )

        fig_weights = px.bar(
            weights_plot_df,
            x="Asset",
            y="Weight",
            color="Portfolio",
            barmode="group",
            title="Portfolio Weights by Asset",
            labels={"Weight": "Weight", "Asset": "Asset"}
        )
        fig_weights.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_weights, use_container_width=True)

        st.subheader("Risk Contribution")
        st.markdown(
            """
Risk contribution shows how much each stock contributes to total portfolio volatility.
A stock can have a small weight but still contribute a large share of risk if it is especially volatile
or strongly linked to the rest of the portfolio.
"""
        )

        rc_col1, rc_col2 = st.columns(2)

        with rc_col1:
            st.markdown("**GMV Risk Contribution**")
            fig_gmv_rc = px.bar(
                gmv_rc_df,
                x="Asset",
                y="Risk Contribution",
                hover_data={"Weight": ":.2%"},
                title="GMV Percentage Risk Contribution"
            )
            fig_gmv_rc.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_gmv_rc, use_container_width=True)

        with rc_col2:
            st.markdown("**Tangency Risk Contribution**")
            fig_tan_rc = px.bar(
                tangency_rc_df,
                x="Asset",
                y="Risk Contribution",
                hover_data={"Weight": ":.2%"},
                title="Tangency Percentage Risk Contribution"
            )
            fig_tan_rc.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_tan_rc, use_container_width=True)

        st.subheader("Efficient Frontier")
        st.markdown(
            """
The efficient frontier shows the set of portfolios with the lowest volatility for each target return.
The Capital Allocation Line (CAL) starts at the risk-free rate and touches the frontier at the tangency portfolio,
which is the highest-Sharpe risky portfolio.
"""
        )

        frontier_fig = go.Figure()

        if not frontier_df.empty:
            frontier_fig.add_trace(go.Scatter(
                x=frontier_df["Volatility"],
                y=frontier_df["Return"],
                mode="lines",
                name="Efficient Frontier"
            ))

        frontier_fig.add_trace(go.Scatter(
            x=cal_x,
            y=cal_y,
            mode="lines",
            name="Capital Allocation Line"
        ))

        frontier_fig.add_trace(go.Scatter(
            x=[eq_vol], y=[eq_ret],
            mode="markers+text",
            name="Equal Weight",
            text=["Equal Weight"],
            textposition="top center",
            marker=dict(size=10)
        ))

        frontier_fig.add_trace(go.Scatter(
            x=[gmv_vol], y=[gmv_ret],
            mode="markers+text",
            name="GMV",
            text=["GMV"],
            textposition="top center",
            marker=dict(size=10)
        ))

        frontier_fig.add_trace(go.Scatter(
            x=[tangency_vol], y=[tangency_ret],
            mode="markers+text",
            name="Tangency",
            text=["Tangency"],
            textposition="top center",
            marker=dict(size=10)
        ))

        frontier_fig.add_trace(go.Scatter(
            x=[custom_vol], y=[custom_ret],
            mode="markers+text",
            name="Custom",
            text=["Custom"],
            textposition="top center",
            marker=dict(size=10)
        ))

        frontier_fig.add_trace(go.Scatter(
            x=[benchmark_vol], y=[benchmark_ret],
            mode="markers+text",
            name="S&P 500",
            text=["S&P 500"],
            textposition="bottom center",
            marker=dict(size=10)
        ))

        stock_ann_returns = asset_returns_only.mean() * TRADING_DAYS
        stock_ann_vols = asset_returns_only.std() * np.sqrt(TRADING_DAYS)

        frontier_fig.add_trace(go.Scatter(
            x=stock_ann_vols.values,
            y=stock_ann_returns.values,
            mode="markers+text",
            name="Individual Stocks",
            text=user_assets,
            textposition="top right",
            marker=dict(size=8)
        ))

        frontier_fig.update_layout(
            title="Efficient Frontier and Capital Allocation Line",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return"
        )
        st.plotly_chart(frontier_fig, use_container_width=True)

        st.subheader("Portfolio Wealth Comparison")
        fig_portfolio_wealth = px.line(
            portfolio_wealth_df,
            x=portfolio_wealth_df.index,
            y=portfolio_wealth_df.columns,
            title="Growth of $10,000: Portfolio Comparison",
            labels={"value": "Portfolio Value ($)", "index": "Date", "variable": "Series"}
        )
        st.plotly_chart(fig_portfolio_wealth, use_container_width=True)

        # =====================================================
    # TAB 6: SENSITIVITY ANALYSIS
    # =====================================================
    with tab6:
        st.subheader("Estimation Window Sensitivity")

        st.markdown(
            """
Mean-variance optimization is sensitive to its inputs. Small changes in the historical sample
used to estimate expected returns and covariances can produce very different optimal weights.
That matters because historical optimization results are only as stable as the inputs that produced them.
This section compares GMV and Tangency portfolios across multiple valid lookback windows.
"""
        )

        available_windows_text = ", ".join(list(lookback_options.keys()))
        st.info(f"Available lookback windows based on your selected data range: {available_windows_text}")

        if sensitivity_results_df.empty:
            st.warning("Sensitivity analysis could not be computed for the available lookback windows.")
        else:
            st.subheader("Sensitivity Comparison Table")
            st.dataframe(
                sensitivity_results_df.style.format({
                    "GMV Annualized Return": "{:.4f}",
                    "GMV Annualized Volatility": "{:.4f}",
                    "Tangency Annualized Return": "{:.4f}",
                    "Tangency Annualized Volatility": "{:.4f}",
                    "Tangency Sharpe Ratio": "{:.4f}"
                }),
                use_container_width=True
            )

            st.subheader("Weights Across Lookback Windows")

            portfolio_choice = st.radio(
                "Select portfolio to visualize",
                options=["GMV", "Tangency"],
                horizontal=True,
                key="sensitivity_portfolio_choice"
            )

            filtered_weights = sensitivity_weights_df[
                sensitivity_weights_df["Portfolio"] == portfolio_choice
            ].copy()

            if not filtered_weights.empty:
                fig_sensitivity_weights = px.bar(
                    filtered_weights,
                    x="Asset",
                    y="Weight",
                    color="Window",
                    barmode="group",
                    title=f"{portfolio_choice} Weights Across Lookback Windows",
                    labels={"Weight": "Weight", "Asset": "Asset"}
                )
                fig_sensitivity_weights.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_sensitivity_weights, use_container_width=True)
            else:
                st.warning(f"No {portfolio_choice} weights were available for the selected lookback windows.")

            st.subheader("Detailed Sensitivity Weights Table")

            weights_pivot = sensitivity_weights_df.pivot_table(
                index=["Portfolio", "Window"],
                columns="Asset",
                values="Weight"
            ).reset_index()

            numeric_weight_cols = [
                col for col in weights_pivot.columns
                if col not in ["Portfolio", "Window"]
            ]

            st.dataframe(
                weights_pivot.style.format(
                    {col: "{:.2%}" for col in numeric_weight_cols}
                ),
                use_container_width=True
            )

st.info("Enter your portfolio inputs in the sidebar and click Run Analysis.")