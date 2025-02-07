import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------------
# Sidebar: Parameter Inputs
# -----------------------------
st.sidebar.title("Prediction Parameters")

lookback_days = st.sidebar.number_input("Lookback Days", min_value=1, max_value=365, value=30, step=1)
current_window_minutes = st.sidebar.number_input("Current Window (Minutes)", min_value=1, max_value=1440, value=60, step=1)
prediction_window_minutes = st.sidebar.number_input("Prediction Window (Minutes)", min_value=1, max_value=1440, value=60, step=1)
k_neighbors = st.sidebar.number_input("K Neighbors", min_value=1, max_value=100, value=20, step=1)
ticker = st.sidebar.text_input("Ticker Symbol", "SPY")

st.title("S&P 500 (SPY) Price Prediction Tool")
st.write("""
This tool leverages historical intraday data to find historical patterns similar to today's recent price behavior and then shows the distribution of subsequent percentage changes.  
Adjust the parameters on the sidebar and the tool will display the predicted outcome distribution.
""")

# -----------------------------
# Function: Download Data (Cached)
# -----------------------------
@st.cache(allow_output_mutation=True)
def get_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    return data

st.write("**Downloading data...**")
data = get_data(ticker, period=f"{lookback_days}d", interval="1m")
if data.empty:
    st.error("No data downloaded. Please check the ticker or your network connection.")
    st.stop()

data.sort_index(inplace=True)
prices = data["Close"]

st.write("**Data Sample:**")
st.dataframe(data.head())

# -----------------------------
# Step 1: Build Historical Samples
# -----------------------------
# For each sample:
#  - The "current window" is defined as the previous CURRENT_WINDOW_MINUTES of data.
#  - The "outcome" is the percentage change from the end of that window to the end of the prediction window.
samples = []
for i in range(current_window_minutes, len(prices) - prediction_window_minutes):
    window_prices = prices.iloc[i - current_window_minutes:i]
    
    # Outcome: Percentage change from the last price in the current window to the price at the end of prediction window.
    current_close = prices.iloc[i - 1]
    future_close = prices.iloc[i + prediction_window_minutes - 1]
    outcome = (future_close - current_close) / current_close * 100

    # --- Feature Engineering ---
    # Feature 1: Percentage change over the current window
    start_price = window_prices.iloc[0]
    end_price = window_prices.iloc[-1]
    feature_pct_change = (end_price - start_price) / start_price * 100

    # Feature 2: Volatility (standard deviation of minute-to-minute returns)
    returns = window_prices.pct_change().dropna()
    feature_volatility = returns.std() * 100  # in percent

    # Feature 3: Slope of price trend (via a linear fit)
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
# Step 2: Nearest Neighbors Model
# -----------------------------
features_cols = ["feature_pct_change", "feature_volatility", "feature_slope"]
X = df_samples[features_cols].values
y = df_samples["outcome"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

nn_model = NearestNeighbors(n_neighbors=k_neighbors)
nn_model.fit(X_scaled)

# -----------------------------
# Step 3: Compute Today's Features
# -----------------------------
current_window_prices = prices.iloc[-current_window_minutes:]
start_price_today = current_window_prices.iloc[0]
end_price_today = current_window_prices.iloc[-1]

feature_pct_change_today = (end_price_today - start_price_today) / start_price_today * 100
returns_today = current_window_prices.pct_change().dropna()
feature_volatility_today = returns_today.std() * 100
x_today = np.arange(len(current_window_prices))
slope_today = np.polyfit(x_today, current_window_prices.values, 1)[0]
feature_slope_today = slope_today / start_price_today * 100

current_features = np.array([feature_pct_change_today, feature_volatility_today, feature_slope_today]).reshape(1, -1)
current_features_scaled = scaler.transform(current_features)

st.subheader("Today's Window Features")
st.write(f"**Percentage Change:** {feature_pct_change_today:.2f}%")
st.write(f"**Volatility:** {feature_volatility_today:.4f}%")
st.write(f"**Slope:** {feature_slope_today:.4f}%")

# -----------------------------
# Step 4: Find Nearest Neighbors & Analyze Outcomes
# -----------------------------
distances, indices = nn_model.kneighbors(current_features_scaled)
neighbors_outcomes = y[indices[0]]

q25 = np.percentile(neighbors_outcomes, 25)
median_outcome = np.median(neighbors_outcomes)
q75 = np.percentile(neighbors_outcomes, 75)
min_outcome = np.min(neighbors_outcomes)
max_outcome = np.max(neighbors_outcomes)

st.subheader("Predicted Outcome Distribution")
st.write(f"**Prediction Window:** Next {prediction_window_minutes} minutes")
st.write(f"**25th percentile:** {q25:.2f}%")
st.write(f"**Median:** {median_outcome:.2f}%")
st.write(f"**75th percentile:** {q75:.2f}%")
st.write(f"**Minimum:** {min_outcome:.2f}%")
st.write(f"**Maximum:** {max_outcome:.2f}%")

# -----------------------------
# Step 5: Visualization
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
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------------
# Sidebar: Parameter Inputs
# -----------------------------
st.sidebar.title("Prediction Parameters")

lookback_days = st.sidebar.number_input("Lookback Days", min_value=1, max_value=365, value=30, step=1)
current_window_minutes = st.sidebar.number_input("Current Window (Minutes)", min_value=1, max_value=1440, value=60, step=1)
prediction_window_minutes = st.sidebar.number_input("Prediction Window (Minutes)", min_value=1, max_value=1440, value=60, step=1)
k_neighbors = st.sidebar.number_input("K Neighbors", min_value=1, max_value=100, value=20, step=1)
ticker = st.sidebar.text_input("Ticker Symbol", "SPY")

st.title("S&P 500 (SPY) Price Prediction Tool")
st.write("""
This tool leverages historical intraday data to find historical patterns similar to today's recent price behavior and then shows the distribution of subsequent percentage changes.  
Adjust the parameters on the sidebar and the tool will display the predicted outcome distribution.
""")

# -----------------------------
# Function: Download Data (Cached)
# -----------------------------
@st.cache(allow_output_mutation=True)
def get_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    return data

st.write("**Downloading data...**")
data = get_data(ticker, period=f"{lookback_days}d", interval="1m")
if data.empty:
    st.error("No data downloaded. Please check the ticker or your network connection.")
    st.stop()

data.sort_index(inplace=True)
prices = data["Close"]

st.write("**Data Sample:**")
st.dataframe(data.head())

# -----------------------------
# Step 1: Build Historical Samples
# -----------------------------
# For each sample:
#  - The "current window" is defined as the previous CURRENT_WINDOW_MINUTES of data.
#  - The "outcome" is the percentage change from the end of that window to the end of the prediction window.
samples = []
for i in range(current_window_minutes, len(prices) - prediction_window_minutes):
    window_prices = prices.iloc[i - current_window_minutes:i]
    
    # Outcome: Percentage change from the last price in the current window to the price at the end of prediction window.
    current_close = prices.iloc[i - 1]
    future_close = prices.iloc[i + prediction_window_minutes - 1]
    outcome = (future_close - current_close) / current_close * 100

    # --- Feature Engineering ---
    # Feature 1: Percentage change over the current window
    start_price = window_prices.iloc[0]
    end_price = window_prices.iloc[-1]
    feature_pct_change = (end_price - start_price) / start_price * 100

    # Feature 2: Volatility (standard deviation of minute-to-minute returns)
    returns = window_prices.pct_change().dropna()
    feature_volatility = returns.std() * 100  # in percent

    # Feature 3: Slope of price trend (via a linear fit)
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
# Step 2: Nearest Neighbors Model
# -----------------------------
features_cols = ["feature_pct_change", "feature_volatility", "feature_slope"]
X = df_samples[features_cols].values
y = df_samples["outcome"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

nn_model = NearestNeighbors(n_neighbors=k_neighbors)
nn_model.fit(X_scaled)

# -----------------------------
# Step 3: Compute Today's Features
# -----------------------------
current_window_prices = prices.iloc[-current_window_minutes:]
start_price_today = current_window_prices.iloc[0]
end_price_today = current_window_prices.iloc[-1]

feature_pct_change_today = (end_price_today - start_price_today) / start_price_today * 100
returns_today = current_window_prices.pct_change().dropna()
feature_volatility_today = returns_today.std() * 100
x_today = np.arange(len(current_window_prices))
slope_today = np.polyfit(x_today, current_window_prices.values, 1)[0]
feature_slope_today = slope_today / start_price_today * 100

current_features = np.array([feature_pct_change_today, feature_volatility_today, feature_slope_today]).reshape(1, -1)
current_features_scaled = scaler.transform(current_features)

st.subheader("Today's Window Features")
st.write(f"**Percentage Change:** {feature_pct_change_today:.2f}%")
st.write(f"**Volatility:** {feature_volatility_today:.4f}%")
st.write(f"**Slope:** {feature_slope_today:.4f}%")

# -----------------------------
# Step 4: Find Nearest Neighbors & Analyze Outcomes
# -----------------------------
distances, indices = nn_model.kneighbors(current_features_scaled)
neighbors_outcomes = y[indices[0]]

q25 = np.percentile(neighbors_outcomes, 25)
median_outcome = np.median(neighbors_outcomes)
q75 = np.percentile(neighbors_outcomes, 75)
min_outcome = np.min(neighbors_outcomes)
max_outcome = np.max(neighbors_outcomes)

st.subheader("Predicted Outcome Distribution")
st.write(f"**Prediction Window:** Next {prediction_window_minutes} minutes")
st.write(f"**25th percentile:** {q25:.2f}%")
st.write(f"**Median:** {median_outcome:.2f}%")
st.write(f"**75th percentile:** {q75:.2f}%")
st.write(f"**Minimum:** {min_outcome:.2f}%")
st.write(f"**Maximum:** {max_outcome:.2f}%")

# -----------------------------
# Step 5: Visualization
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
