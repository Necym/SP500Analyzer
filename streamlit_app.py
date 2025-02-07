import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =============================
# Sidebar: Parameter Inputs
# =============================
st.sidebar.title("Prediction Parameters")

# Historical lookback period (for training samples)
lookback_days = st.sidebar.number_input("Lookback Days (for historical data)", min_value=1, max_value=365, value=7, step=1)
# Note: For 1-minute data, a shorter lookback period (e.g. 7 days) is often more reliable.

# Prediction window length (in minutes)
prediction_window_minutes = st.sidebar.number_input("Prediction Window (Minutes)", min_value=1, max_value=1440, value=60, step=1)

# Number of nearest neighbors to use in the prediction
k_neighbors = st.sidebar.number_input("K Neighbors", min_value=1, max_value=100, value=20, step=1)

# Ticker symbol (default SPY)
ticker = st.sidebar.text_input("Ticker Symbol", "SPY")

# Option: use a custom current window rather than the most recent window
use_custom_window = st.sidebar.checkbox("Use Custom Current Window?", value=False)
if use_custom_window:
    custom_date = st.sidebar.date_input("Select Date for Custom Window", value=pd.to_datetime("today").date(), key="custom_date")
    custom_start_time = st.sidebar.time_input("Custom Start Time", value=pd.to_datetime("09:30").time(), key="custom_start")
    custom_end_time = st.sidebar.time_input("Custom End Time", value=pd.to_datetime("10:30").time(), key="custom_end")
    # Combine date and times to form timestamps
    custom_start_dt = pd.Timestamp.combine(custom_date, custom_start_time)
    custom_end_dt = pd.Timestamp.combine(custom_date, custom_end_time)
    # Compute the window length in minutes
    computed_window_minutes = int((custom_end_dt - custom_start_dt).total_seconds() / 60)
    st.sidebar.write(f"Custom window length: {computed_window_minutes} minutes")
else:
    # If not using a custom window, specify the length in minutes to take from the most recent data
    current_window_minutes = st.sidebar.number_input("Current Window (Minutes)", min_value=1, max_value=1440, value=60, step=1)
    computed_window_minutes = current_window_minutes

# Add a note about intraday data availability
st.sidebar.info("Note: 1-minute interval data is often only available for recent trading days. "
                "Ensure that your selected dates correspond to days when the market was open.")

# Button to trigger the analysis
run_analysis = st.sidebar.button("Run Analysis")

# =============================
# Main App Title and Description
# =============================
st.title("S&P 500 (SPY) Price Prediction Tool")
st.write("""
This tool leverages historical intraday data to find patterns similar to a selected current window and then shows the distribution of subsequent percentage changes.  
Configure your settings on the sidebar and click **Run Analysis** to fetch the data and perform the analysis.
""")

# Only proceed when Run Analysis is clicked.
if run_analysis:
    # -----------------------------
    # Function: Download Data (Cached)
    # -----------------------------
    @st.cache_data
    def get_data(ticker, period, interval):
        data = yf.download(ticker, period=period, interval=interval)
        return data

    st.write("**Downloading data...**")
    data = get_data(ticker, period=f"{lookback_days}d", interval="1m")
    if data.empty:
        st.error("No data downloaded. Please check the ticker, ensure the market was open on the selected dates, or try a shorter lookback period.")
        st.stop()

    data.sort_index(inplace=True)
    prices = data["Close"]

    st.write("**Data Sample:**")
    st.dataframe(data.head())

    # -----------------------------
    # Extract the "Current" Window Data
    # -----------------------------
    if use_custom_window:
        # Use the custom window specified by the user.
        # Check if the data's index is timezone-aware and localize if needed.
        tz_info = data.index.tz
        if tz_info is not None:
            custom_start_dt = custom_start_dt.tz_localize(tz_info)
            custom_end_dt = custom_end_dt.tz_localize(tz_info)
        
        current_window = data.loc[custom_start_dt:custom_end_dt]["Close"]
        if current_window.empty:
            st.error("No data available for the specified custom window. Please adjust the date/time inputs to a trading day/time.")
            st.stop()
    else:
        # Otherwise, use the most recent computed_window_minutes of data
        current_window = prices.iloc[-computed_window_minutes:]
    
    # -----------------------------
    # Build Historical Samples
    # -----------------------------
    # Each sample uses a "historical window" of length computed_window_minutes.
    # The outcome is the percentage change from the end of that window to prediction_window_minutes later.
    historical_window_minutes = computed_window_minutes
    samples = []
    for i in range(historical_window_minutes, len(prices) - prediction_window_minutes):
        window_prices = prices.iloc[i - historical_window_minutes:i]
        current_close = prices.iloc[i - 1]
        future_close = prices.iloc[i + prediction_window_minutes - 1]
        outcome = (future_close - current_close) / current_close * 100

        # Feature 1: Percentage change over the window
        start_price = window_prices.iloc[0]
        end_price = window_prices.iloc[-1]
        feature_pct_change = (end_price - start_price) / start_price * 100

        # Feature 2: Volatility (std. dev. of minute-to-minute returns)
        returns = window_prices.pct_change().dropna()
        feature_volatility = returns.std() * 100

        # Feature 3: Slope of the price trend (via linear regression)
        x = np.arange(len(window_prices))
        slope = np.polyfit(x, window_prices.values, 1)[0]
        feature_slope = slope / start_price * 100

        samples.append({
            "feature_pct_change": feature_pct_change,
            "feature_volatility": feature_volatility,
            "feature_slope": feature_slope,
            "outcome": outcome
        })

    df_samples = pd.DataFrame(samples)
    st.write("**Historical Samples (Feature Engineering):**")
    st.dataframe(df_samples.head())

    # -----------------------------
    # Build the Nearest Neighbors Model
    # -----------------------------
    features_cols = ["feature_pct_change", "feature_volatility", "feature_slope"]
    X = df_samples[features_cols].values
    y = df_samples["outcome"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn_model = NearestNeighbors(n_neighbors=k_neighbors)
    nn_model.fit(X_scaled)

    # -----------------------------
    # Compute Features for the Current Window
    # -----------------------------
    current_window_prices = current_window
    if len(current_window_prices) < 2:
        st.error("Not enough data in the current window to compute features.")
        st.stop()

    start_price_today = current_window_prices.iloc[0]
    end_price_today = current_window_prices.iloc[-1]
    feature_pct_change_today = (end_price_today - start_price_today) / start_price_today * 100
    returns_today = current_window_prices.pct_change().dropna()
    feature_volatility_today = returns_today.std() * 100
    x_today = np.arange(len(current_window_prices))
    slope_today = np.polyfit(x_today, current_window_prices.values, 1)[0]
    feature_slope_today = slope_today / start_price_today * 100

    # Explicitly cast to float if needed
    current_features = np.array([
        float(feature_pct_change_today),
        float(feature_volatility_today),
        float(feature_slope_today)
    ]).reshape(1, -1)
    current_features_scaled = scaler.transform(current_features)

    st.subheader("Current Window Features")
    st.write(f"**Percentage Change:** {float(feature_pct_change_today):.2f}%")
    st.write(f"**Volatility:** {float(feature_volatility_today):.4f}%")
    st.write(f"**Slope:** {float(feature_slope_today):.4f}%")

    # -----------------------------
    # Find Nearest Neighbors & Analyze Outcomes
    # -----------------------------

    distances, indices = nn_model.kneighbors(current_features_scaled)
    # Ensure neighbors_outcomes is a numpy array
    neighbors_outcomes = np.array(y[indices[0]])

    q25 = np.percentile(neighbors_outcomes, 25)
    median_outcome = np.median(neighbors_outcomes)
    q75 = np.percentile(neighbors_outcomes, 75)
    min_outcome = np.min(neighbors_outcomes)
    max_outcome = np.max(neighbors_outcomes)

    st.subheader("Predicted Outcome Distribution")
    st.write(f"**Prediction Window:** Next {prediction_window_minutes} minutes")
    st.write(f"**25th Percentile:** {q25:.2f}%")
    st.write(f"**Median:** {median_outcome:.2f}%")
    st.write(f"**75th Percentile:** {q75:.2f}%")
    st.write(f"**Minimum:** {min_outcome:.2f}%")
    st.write(f"**Maximum:** {max_outcome:.2f}%")

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(neighbors_outcomes, bins=10, color="skyblue", edgecolor="black")
    ax.axvline(q25, color="red", linestyle="dashed", linewidth=1, label="25th Percentile")
    ax.axvline(median_outcome, color="green", linestyle="dashed", linewidth=1, label="Median")
    ax.axvline(q75, color="red", linestyle="dashed", linewidth=1, label="75th Percentile")
    ax.set_title(f"Outcome Distribution for Next {prediction_window_minutes} Minutes")
    ax.set_xlabel("Percentage Change (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
