"""Streamlit dashboard for historical market data and model evaluation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from local_analysis import analyze_symbol
from market_data import get_stock_data, market_profile
from paper_trading import order_history, place_paper_order, positions
from research_store import recent_experiments

st.set_page_config(page_title="GreenToBuy | Market Analytics", page_icon="📈", layout="wide")
TICKERS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon", "GOOGL": "Alphabet",
    "META": "Meta Platforms", "TSLA": "Tesla", "AVGO": "Broadcom", "BRK-B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase", "LLY": "Eli Lilly", "V": "Visa", "XOM": "Exxon Mobil", "MA": "Mastercard",
    "COST": "Costco", "WMT": "Walmart", "ORCL": "Oracle", "NFLX": "Netflix", "AMD": "AMD",
    "CRM": "Salesforce", "QCOM": "Qualcomm", "ADBE": "Adobe", "INTC": "Intel", "CSCO": "Cisco",
    "IBM": "IBM", "GE": "GE Aerospace", "CAT": "Caterpillar", "BA": "Boeing", "DIS": "Walt Disney",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "NKE": "Nike", "MCD": "McDonald's", "UBER": "Uber",
    "PLTR": "Palantir", "COIN": "Coinbase", "SHOP": "Shopify", "PYPL": "PayPal", "BABA": "Alibaba",
    "TSM": "Taiwan Semiconductor", "SONY": "Sony", "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ ETF",
    "DIA": "SPDR Dow Jones ETF", "IWM": "iShares Russell 2000 ETF", "VTI": "Vanguard Total Market ETF",
    "VOO": "Vanguard S&P 500 ETF", "GLD": "SPDR Gold Shares", "SLV": "iShares Silver Trust",
    "TLT": "iShares 20+ Year Treasury ETF", "XLF": "Financial Select Sector ETF",
    "XLE": "Energy Select Sector ETF", "XAUUSD": "Gold / US Dollar", "EURUSD": "Euro / US Dollar",
    "GBPUSD": "British Pound / US Dollar",
    "RELIANCE.NS": "Reliance Industries · NSE", "TCS.NS": "Tata Consultancy Services · NSE",
    "HDFCBANK.NS": "HDFC Bank · NSE", "ICICIBANK.NS": "ICICI Bank · NSE", "INFY.NS": "Infosys · NSE",
    "SBIN.NS": "State Bank of India · NSE", "BHARTIARTL.NS": "Bharti Airtel · NSE",
    "ITC.NS": "ITC Limited · NSE", "LT.NS": "Larsen & Toubro · NSE", "HINDUNILVR.NS": "Hindustan Unilever · NSE",
    "AXISBANK.NS": "Axis Bank · NSE", "KOTAKBANK.NS": "Kotak Mahindra Bank · NSE",
    "MARUTI.NS": "Maruti Suzuki · NSE", "TATAMOTORS.NS": "Tata Motors · NSE",
    "TATASTEEL.NS": "Tata Steel · NSE", "SUNPHARMA.NS": "Sun Pharmaceutical · NSE",
    "BAJFINANCE.NS": "Bajaj Finance · NSE", "ASIANPAINT.NS": "Asian Paints · NSE",
    "TITAN.NS": "Titan Company · NSE", "ADANIENT.NS": "Adani Enterprises · NSE",
    "NTPC.NS": "NTPC · NSE", "POWERGRID.NS": "Power Grid · NSE", "ONGC.NS": "ONGC · NSE",
    "WIPRO.NS": "Wipro · NSE", "HCLTECH.NS": "HCL Technologies · NSE",
    "ULTRACEMCO.NS": "UltraTech Cement · NSE", "^NSEI": "NIFTY 50 Index",
    "^NSEBANK": "NIFTY Bank Index", "^BSESN": "BSE SENSEX Index",
}

st.markdown("""
<style>
  :root { color-scheme: dark; }
  .stApp { background: #080d13; color: #e9eef5; }
  [data-testid="stSidebar"] { background: #0d151e; border-right: 1px solid #1d2a37; }
  [data-testid="stMetric"] { background: linear-gradient(145deg,#111c27,#0d151e); border: 1px solid #223242; padding: 15px 17px; border-radius: 14px; }
  [data-testid="stMetricLabel"] { color: #9aabba; }
  h1, h2, h3 { letter-spacing: -0.035em; }
  h1 { font-size: 2.25rem !important; }
  [data-testid="stTabs"] button { font-weight: 600; }
  .gtb-eyebrow { color: #50d7a2; text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 700; }
  .gtb-muted { color: #93a1af; font-size: .88rem; }
  .gtb-pill { border: 1px solid #244535; background: #10251c; color: #6de4ad; border-radius: 999px; padding: 6px 11px; font-size: .78rem; }
</style>
""", unsafe_allow_html=True)

st.title("GreenToBuy")
st.caption("Market research workspace · forecasts, risk, and model evidence in one view")

with st.sidebar:
    st.markdown('<div class="gtb-eyebrow">Workspace</div>', unsafe_allow_html=True)
    st.subheader("Choose a market")
    ticker_options = list(TICKERS)
    selected_ticker = st.selectbox(
        "Ticker or index",
        ticker_options,
        index=ticker_options.index("AAPL"),
        format_func=lambda ticker: f"{ticker}  ·  {TICKERS[ticker]}",
        help="Search by symbol or company name, or enter another Yahoo Finance symbol below.",
    )
    custom_ticker = st.text_input("Custom symbol", placeholder="e.g. MSFT, RELIANCE.NS, EURUSD", max_chars=20).strip().upper()
    symbol = custom_ticker or selected_ticker
    with st.form("analysis_controls"):
        interval = st.selectbox("Bar interval", ["daily", "60min", "15min", "5min"])
        horizon_unit = "trading day" if interval == "daily" else "bar"
        horizon = st.selectbox("Forecast horizon", [1, 3, 5, 10, 20, 30], format_func=lambda x: f"{x} {horizon_unit if x == 1 else horizon_unit + 's'}")
        model_labels = {
            "auto": "Validated all-family model",
            "weighted ensemble": "Stacked ensemble",
            "ridge": "Ridge regression",
            "random forest": "Random forest",
            "extra trees": "Extra Trees",
            "hist gradient boosting": "Hist gradient boosting",
            "svr": "RBF SVR (approximate)",
        }
        model = st.selectbox("Forecast method", list(model_labels), format_func=lambda value: model_labels[value])
        analyze = st.form_submit_button("Run market analysis", type="primary", use_container_width=True)
    st.caption("Data source: Yahoo Finance · local research history · no broker orders")

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_latest_quote(ticker: str) -> dict:
    data = get_stock_data(ticker, "5min", force_refresh=True)
    timestamp = data.index[-1]
    return {"symbol": ticker, "price": float(data["close"].iloc[-1]), "timestamp": timestamp.isoformat(), "currency": "INR" if market_profile(ticker) == "india" else None}


if analyze:
    with st.spinner(f"Fetching {symbol} data and validating the forecast…"):
        try:
            st.session_state["analysis_result"] = analyze_symbol(symbol, interval, horizon, model)
            st.session_state.pop("analysis_error", None)
        except Exception as exc:
            st.session_state["analysis_error"] = str(exc)
            st.session_state.pop("analysis_result", None)

if st.session_state.get("analysis_error"):
    st.error(f"Analysis failed: {st.session_state['analysis_error']}")

result = st.session_state.get("analysis_result")
if result is not None:
    required_diagnostics = {
        "trailing_returns_pct",
        "price_vs_moving_average_pct",
        "volume_vs_20_bar_average",
        "price_position_in_20_bar_range_pct",
        "positive_return_frequency_pct",
        "realized_volatility_pct_per_bar",
    }
    required_risk = {
        "historical_var_95_pct_per_bar",
        "historical_cvar_95_pct_per_bar",
        "downside_deviation_pct_per_bar",
    }
    diagnostics = result.get("diagnostics", {})
    risk = result.get("risk", {})
    if not required_diagnostics.issubset(diagnostics) or not required_risk.issubset(risk):
        st.session_state.pop("analysis_result", None)
        st.info("This analysis was created by an older app version. Run the analysis again to load the expanded statistics.")
        st.stop()

if result is None:
    st.info("Choose a market in the sidebar, then run an analysis to load its forecast and research statistics.")
    intro = st.columns(3)
    intro[0].metric("Markets", "Equities · India · Forex")
    intro[1].metric("Validation", "Purged walk-forward")
    intro[2].metric("Research", "Saved locally")
    st.caption("The model uses historical data and validation evidence. Forecasts are uncertain estimates, not guaranteed outcomes.")
    st.stop()

forecast = result["forecast"]
diagnostics = result["diagnostics"]
risk = result["risk"]
symbol = result["symbol"]
interval = result["interval"]
horizon = result["horizon"]
unit = "trading day" if interval == "daily" else "bar"
current = result["latest_price"]["price"]
expected = forecast["return_forecast_pct"]

head_cols = st.columns([3, 1])
head_cols[0].markdown(f"### {symbol} <span class='gtb-muted'>· {result['market_profile'].replace('_', ' ').title()} · {interval}</span>", unsafe_allow_html=True)
head_cols[0].caption(f"Latest bar {result['latest_price']['timestamp']} · {result['data_source']}")
head_cols[1].markdown(f"<div style='text-align:right;padding-top:16px'><span class='gtb-pill'>{diagnostics['regime']}</span></div>", unsafe_allow_html=True)

baseline_mae = forecast["naive_metrics"]["mae"]
selected_mae = forecast["metrics"]["mae"]
holdout_skill = ((baseline_mae - selected_mae) / baseline_mae * 100) if baseline_mae > 0 else 0.0
band = forecast["uncertainty"]["intervals"]["90"]
summary = st.columns(3)
summary[0].metric("Latest price", f"{current:,.2f}")
summary[1].metric(f"{horizon}-{unit} forecast", f"{forecast['forecast']:,.2f}", f"{expected:+.2f}%")
summary[2].metric("Empirical 90% range", f"{band['lower']:,.2f} – {band['upper']:,.2f}", help="Range based on past selection-fold errors; future prices can fall outside it.")
st.caption(f"Model: {forecast['model']} · Holdout MAE {selected_mae:,.3f} · Forecasts are uncertain and cannot be 100% predictable.")

if forecast["model"] == "Naïve: unchanged close":
    st.warning("No candidate cleared the validation hurdle. The forecast is the unchanged-close benchmark.")
elif holdout_skill < 0:
    st.warning("The selected model underperformed the unchanged-close benchmark on the separate holdout.")

overview_tab, model_tab, risk_tab, history_tab, paper_tab = st.tabs(["Overview", "Model lab", "Risk & indicators", "Saved research", "Paper trading"])

with overview_tab:
    external_data = result.get("external_data", {})
    if result["market_profile"] == "india":
        if external_data.get("available"):
            change = external_data.get("usd_inr_return_5obs_pct")
            move = f" · 5-observation move {change:+.2f}%" if change is not None else ""
            st.caption(f"Model macro input: USD/INR {external_data['usd_inr_rate']:.4f} · reference {external_data['reference_date']} · available {external_data['available_from']} · one-day lag{move} · Frankfurter")
        else:
            st.caption("Optional USD/INR macro input unavailable; this forecast used price-derived features only.")
    chart = pd.DataFrame(result["chart_data"])
    chart["x"] = pd.to_datetime(chart["x"], utc=True)
    chart["sma20"] = chart["c"].rolling(20).mean()
    chart["sma50"] = chart["c"].rolling(50).mean()
    chart_head, chart_control = st.columns([4, 1])
    chart_head.subheader("Price history and forecast")
    show_sma = chart_control.toggle("Trend lines", value=True)
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.78, 0.22])
    figure.add_trace(go.Candlestick(x=chart.x, open=chart.o, high=chart.h, low=chart.l, close=chart.c, name=symbol), row=1, col=1)
    if show_sma:
        figure.add_trace(go.Scatter(x=chart.x, y=chart.sma20, mode="lines", name="SMA 20", line={"color": "#55d6a5", "width": 1.3}), row=1, col=1)
        figure.add_trace(go.Scatter(x=chart.x, y=chart.sma50, mode="lines", name="SMA 50", line={"color": "#f1bb66", "width": 1.3}), row=1, col=1)
    figure.add_trace(go.Bar(x=chart.x, y=chart.v, name="Volume", marker_color="#42627b", opacity=0.7), row=2, col=1)
    last_time = chart.x.iloc[-1]
    future_time = (pd.bdate_range(start=last_time.tz_localize(None), periods=horizon + 1)[-1].tz_localize("UTC") if interval == "daily" else last_time + pd.Timedelta(minutes={"5min": 5, "15min": 15, "60min": 60}[interval] * horizon))
    figure.add_trace(go.Scatter(x=[last_time, future_time], y=[current, forecast["forecast"]], mode="lines+markers", name="Forecast", line={"color": "#45d99d", "dash": "dash", "width": 2.5}), row=1, col=1)
    figure.add_trace(go.Scatter(x=[future_time, future_time], y=[band["lower"], band["upper"]], mode="lines+markers", name="Empirical 90% band", line={"color": "#edbd68", "width": 5}), row=1, col=1)
    figure.update_layout(template="plotly_dark", height=590, margin={"l": 8, "r": 8, "t": 16, "b": 8}, xaxis_rangeslider_visible=False, legend={"orientation": "h", "y": 1.04}, hovermode="x unified")
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(figure, use_container_width=True)

    with st.expander("Historical statistics", expanded=False):
        st.caption("These summarize past bars and do not guarantee future performance.")
        returns = diagnostics["trailing_returns_pct"]
        snapshot = st.columns(5)
        for column, period in zip(snapshot, ("1", "5", "20", "60", "252")):
            value = returns.get(period)
            column.metric(f"Return · {period} bars", f"{value:+.2f}%" if value is not None else "N/A")
        more = st.columns(3)
        more[0].metric("Price vs SMA 20 / 50", f"{diagnostics['price_vs_moving_average_pct']['20']:+.2f}% / {diagnostics['price_vs_moving_average_pct']['50']:+.2f}%" if diagnostics["price_vs_moving_average_pct"]["20"] is not None and diagnostics["price_vs_moving_average_pct"]["50"] is not None else "N/A")
        more[1].metric("Volume vs 20-bar average", f"{diagnostics['volume_vs_20_bar_average']:.2f}×" if diagnostics["volume_vs_20_bar_average"] is not None else "N/A")
        more[2].metric("Position in 20-bar range", f"{diagnostics['price_position_in_20_bar_range_pct']:.0f}%" if diagnostics["price_position_in_20_bar_range_pct"] is not None else "N/A")
        more = st.columns(2)
        more[0].metric("Positive bars · 20", f"{diagnostics['positive_return_frequency_pct']['20']:.0f}%" if diagnostics["positive_return_frequency_pct"]["20"] is not None else "N/A")
        more[1].metric("Positive bars · 60", f"{diagnostics['positive_return_frequency_pct']['60']:.0f}%" if diagnostics["positive_return_frequency_pct"]["60"] is not None else "N/A")

with model_tab:
    st.subheader("Holdout scorecard")
    st.caption(f"{forecast['validation_points']} separate chronological predictions · through {forecast['validation_end']}")
    score_rows = []
    for name, values in forecast["model_comparison"].items():
        score_rows.append({
            "Method": name,
            "Selection MAE (% return)": values["selection_mae"],
            "Holdout MAE": values["mae"],
            "RMSE": values["rmse"],
            "MAPE (%)": values["mape"],
            "R²": values["r2"],
            "Direction (%)": values["directional_accuracy"],
            "Fit time (s)": values["training_seconds"],
        })
    score_frame = pd.DataFrame(score_rows).set_index("Method").sort_values("Selection MAE (% return)")
    st.dataframe(score_frame.style.format({column: "{:.3f}" for column in score_frame.columns}), use_container_width=True)
    skill_cols = st.columns(4)
    skill_cols[0].metric("Selected vs naive MAE", f"{holdout_skill:+.2f}%")
    skill_cols[1].metric("Naive holdout MAE", f"{baseline_mae:,.3f}")
    skill_cols[2].metric("Holdout direction", f"{forecast['metrics']['directional_accuracy']:.1f}%")
    skill_cols[3].metric("90% interval coverage", f"{band['holdout_coverage_pct']:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Actual vs forecast")
        holdout = pd.DataFrame(forecast["validation_predictions"])
        holdout["timestamp"] = pd.to_datetime(holdout["timestamp"], utc=True)
        actual_fig = go.Figure()
        actual_fig.add_trace(go.Scatter(x=holdout.timestamp, y=holdout.actual, mode="lines+markers", name="Actual", line={"color": "#e7edf4"}))
        actual_fig.add_trace(go.Scatter(x=holdout.timestamp, y=holdout.predicted, mode="lines+markers", name="Selected", line={"color": "#45d99d"}))
        actual_fig.add_trace(go.Scatter(x=holdout.timestamp, y=holdout.unchanged_close, mode="lines", name="Unchanged close", line={"color": "#edbd68", "dash": "dot"}))
        actual_fig.update_layout(template="plotly_dark", height=350, margin={"l": 8, "r": 8, "t": 8, "b": 8}, hovermode="x unified")
        st.plotly_chart(actual_fig, use_container_width=True)
    with right:
        st.markdown("#### Ensemble weights")
        if forecast["ensemble_weights"]:
            weights = pd.Series(forecast["ensemble_weights"]).sort_values()
            weights_fig = go.Figure(go.Bar(x=weights.values * 100, y=weights.index, orientation="h", marker_color="#45d99d", text=[f"{weight:.1%}" for weight in weights.values], textposition="outside"))
            weights_fig.update_layout(template="plotly_dark", height=350, xaxis_title="Weight (%)", margin={"l": 8, "r": 35, "t": 8, "b": 8})
            st.plotly_chart(weights_fig, use_container_width=True)
        else:
            st.info("Weights are shown for Auto and Stacked ensemble runs.")
    st.info("Model weights and selection scores come from earlier chronological folds. The displayed holdout was not used to fit the ensemble.")

with risk_tab:
    st.subheader("Risk profile")
    risk_row = st.columns(6)
    risk_row[0].metric("EWMA vol / bar", f"{risk['ewma_volatility_pct_per_bar']:.3f}%")
    risk_row[1].metric(f"EWMA vol / {horizon} bars", f"{risk['sqrt_time_horizon_volatility_pct']:.3f}%")
    risk_row[2].metric("Historical VaR · 95% / bar", f"{risk['historical_var_95_pct_per_bar']:.3f}%")
    risk_row[3].metric("Historical CVaR · 95% / bar", f"{risk['historical_cvar_95_pct_per_bar']:.3f}%")
    risk_row[4].metric("Max drawdown · 252 bars", f"{diagnostics['max_drawdown_last_252_bars_pct']:.2f}%")
    risk_row[5].metric("Downside deviation / bar", f"{risk['downside_deviation_pct_per_bar']:.3f}%")
    st.caption(risk["range_assumption"] + " Historical VaR/CVaR use the trailing return sample and are not maximum-loss estimates.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Volatility estimators")
        vol = {
            "Rolling 20-bar": diagnostics["realized_volatility_pct_per_bar"]["20"],
            "Rolling 60-bar": diagnostics["realized_volatility_pct_per_bar"]["60"],
            "EWMA": risk["ewma_volatility_pct_per_bar"],
            "Parkinson": risk["parkinson_volatility_pct_per_bar"],
            "Garman–Klass": risk["garman_klass_volatility_pct_per_bar"],
        }
        vol_frame = pd.DataFrame({"Estimator": list(vol), "Volatility (% / bar)": list(vol.values())}).dropna()
        vol_fig = go.Figure(go.Bar(x=vol_frame["Estimator"], y=vol_frame["Volatility (% / bar)"], marker_color="#edbd68"))
        vol_fig.update_layout(template="plotly_dark", height=330, margin={"l": 8, "r": 8, "t": 8, "b": 8})
        st.plotly_chart(vol_fig, use_container_width=True)
    with right:
        st.markdown("#### Trend and return distribution")
        closes = chart["c"].astype(float)
        bar_returns = closes.pct_change().dropna() * 100
        distribution = go.Figure(go.Histogram(x=bar_returns, nbinsx=30, marker_color="#5da8d6", opacity=0.85, name="Bar returns"))
        distribution.update_layout(template="plotly_dark", height=330, xaxis_title="Return (%)", yaxis_title="Bars", margin={"l": 8, "r": 8, "t": 8, "b": 8})
        st.plotly_chart(distribution, use_container_width=True)

    stats_cols = st.columns(4)
    stats_cols[0].metric("Hurst estimate", f"{diagnostics['hurst_estimate']:.3f}" if diagnostics["hurst_estimate"] is not None else "Insufficient history")
    stats_cols[1].metric("Lag-1 return autocorrelation", f"{diagnostics['lag1_return_autocorrelation']:.3f}" if diagnostics["lag1_return_autocorrelation"] is not None else "N/A")
    stats_cols[2].metric("Return skew", f"{diagnostics['return_skew']:.3f}" if diagnostics["return_skew"] is not None else "N/A")
    stats_cols[3].metric("Excess kurtosis", f"{diagnostics['return_excess_kurtosis']:.3f}" if diagnostics["return_excess_kurtosis"] is not None else "N/A")
    st.markdown("#### Indicator readings")
    indicators = result["indicators"]
    indicator_labels = {"sma_20": "SMA 20", "sma_50": "SMA 50", "sma_200": "SMA 200", "rsi": "RSI 14", "macd": "MACD", "atr": "ATR 14", "adx": "ADX 14", "bb_upper": "Bollinger upper", "bb_lower": "Bollinger lower"}
    indicator_frame = pd.DataFrame([{"Indicator": label, "Value": indicators[key]} for key, label in indicator_labels.items()]).set_index("Indicator")
    st.dataframe(indicator_frame.style.format({"Value": "{:.4f}"}), use_container_width=True)

with history_tab:
    st.subheader("Saved research runs")
    runs = recent_experiments(symbol, limit=100)
    if not runs:
        st.info("No saved runs for this symbol yet. Run an analysis to create the first entry.")
    else:
        history = pd.DataFrame([{
            "Run ID": run["experiment_id"],
            "Created (UTC)": run["created_at"],
            "Interval": run["interval"],
            "Horizon": run["horizon"],
            "Model": run["selected_model"],
            "Reference": run["reference_price"],
            "Forecast": run["forecast_price"],
            "Return (%)": run["return_forecast_pct"],
            "Holdout MAE": run["metrics"]["selected"]["mae"],
            "Naive MAE": run["metrics"]["naive"]["mae"],
        } for run in runs])
        st.dataframe(history, hide_index=True, use_container_width=True)
        chosen_run = st.selectbox("Inspect run", history["Run ID"].tolist(), format_func=lambda value: f"{value[:8]} · {history.loc[history['Run ID'] == value, 'Created (UTC)'].iloc[0]}")
        detail = next(run for run in runs if run["experiment_id"] == chosen_run)
        detail_cols = st.columns(3)
        detail_cols[0].metric("Selected holdout skill", f"{((detail['metrics']['naive']['mae'] - detail['metrics']['selected']['mae']) / detail['metrics']['naive']['mae'] * 100):+.2f}%" if detail["metrics"]["naive"]["mae"] else "N/A")
        detail_cols[1].metric("Training time", f"{detail['training_seconds']:.2f}s")
        detail_cols[2].metric("Features", str(len(detail["config"]["features"])))
        with st.expander("Run configuration and metrics"):
            st.json({"config": detail["config"], "metrics": detail["metrics"], "diagnostics": detail["diagnostics"]})

with paper_tab:
    st.subheader("Paper trading")
    st.caption("Simulated local fills only. Yahoo Finance five-minute bars can be delayed; no broker order is placed.")
    refresh_col, quote_col = st.columns([1, 3])
    if refresh_col.button("Refresh Yahoo quote", key="refresh_paper_quote"):
        _fetch_latest_quote.clear()
        try:
            st.session_state["paper_quote"] = _fetch_latest_quote(symbol)
            st.session_state.pop("paper_quote_error", None)
        except Exception as exc:
            st.session_state["paper_quote_error"] = str(exc)
    quote = st.session_state.get("paper_quote")
    if quote and quote.get("symbol") == symbol:
        quote_time = pd.Timestamp(quote["timestamp"])
        age_seconds = max(0, (pd.Timestamp.now(tz="UTC") - quote_time).total_seconds())
        quote["status"] = "live" if age_seconds <= 60 else "stale"
        quote_col.metric("Latest bar close", f"{quote['price']:,.5f}".rstrip("0").rstrip("."), help=f"Bar timestamp: {quote['timestamp']} · age {age_seconds:.0f}s")
        quote_col.caption(f"Bar time {quote['timestamp']} · age {age_seconds:.0f}s")
    elif st.session_state.get("paper_quote_error"):
        st.error(f"Quote unavailable: {st.session_state['paper_quote_error']}")
    else:
        quote_col.info("Refresh a quote before placing a paper order.")

    if symbol.startswith("^"):
        st.info("Index instruments are view-only in paper mode.")
    elif quote and quote.get("symbol") == symbol:
        with st.form("paper_order_form"):
            order_cols = st.columns(3)
            side = order_cols[0].selectbox("Side", ["BUY", "SELL"])
            quantity = order_cols[1].number_input("Quantity", min_value=0.000001, value=1.0)
            submit_order = order_cols[2].form_submit_button("Place paper order", type="primary")
        if submit_order:
            try:
                order = place_paper_order(symbol, side, quantity, quote)
                st.success(f"Paper {side.lower()} filled at {order['fill_price']:,.5f}; no broker order was sent.")
            except ValueError as exc:
                st.error(str(exc))

    positions_frame = pd.DataFrame(positions())
    if not positions_frame.empty:
        st.markdown("#### Open positions")
        st.dataframe(positions_frame, hide_index=True, use_container_width=True)
    orders_frame = pd.DataFrame(order_history(50))
    if not orders_frame.empty:
        st.markdown("#### Recent paper orders")
        st.dataframe(orders_frame, hide_index=True, use_container_width=True)
