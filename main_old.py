import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def get_ticker_data(ticker_symbol, start_date, end_date):
    """Fetches historical stock data."""
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data

def create_sequences(data, time_steps):
    """Creates sequences from the time-series data."""
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
    return np.array(sequences)

def create_lstm_autoencoder(timesteps, features):
    """Creates and compiles an LSTM autoencoder model."""
    model = Sequential()
    model.add(LSTM(128, input_shape=(timesteps, features)))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(features)))
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_loss_distribution(train_loss, test_loss, threshold):
    """Plots the distribution of reconstruction errors."""
    plt.figure(figsize=(12, 6))
    sns.histplot(train_loss, bins=50, kde=True, label='Train Loss', color='blue')
    sns.histplot(test_loss, bins=50, kde=True, label='Test Loss', color='orange')
    plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Distribution of Reconstruction Loss')
    plt.xlabel('Mean Absolute Error (Loss)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def plot_anomalies(data, anomalies, ticker_symbol):
    """Plots the stock data and highlights the detected anomalies."""
    plt.figure(figsize=(15, 7))
    plt.plot(data.index, data['Close'], label='Close Price', alpha=0.8)
    if not anomalies.empty:
        plt.scatter(anomalies.index, anomalies['Close'], color='red', label='Anomaly', s=60, zorder=5)
    plt.title(f'{ticker_symbol} Stock Price with Detected Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def main():
    # --- 1. Fetch Data ---
    TICKER = 'TSLA'
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    TIME_STEPS = 30
    
    df = get_ticker_data(TICKER, START_DATE, END_DATE)
    data = df[['Close']].values

    # --- 2. Preprocess Data ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    sequences = create_sequences(scaled_data, TIME_STEPS)

    train_size = int(len(sequences) * 0.8)
    train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]

    # --- 3. Build and Train the LSTM Autoencoder ---
    model = create_lstm_autoencoder(TIME_STEPS, features=1)
    history = model.fit(train_sequences, train_sequences, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    # --- 4. Detect Anomalies ---
    train_mae_loss = np.mean(np.abs(model.predict(train_sequences, verbose=0) - train_sequences), axis=1)
    test_mae_loss = np.mean(np.abs(model.predict(test_sequences, verbose=0) - test_sequences), axis=1)

    threshold = np.quantile(train_mae_loss.flatten(), 0.95)
    print(f"\nAnomaly Threshold (95th percentile of train loss): {threshold:.4f}")

    plot_loss_distribution(train_mae_loss.flatten(), test_mae_loss.flatten(), threshold)

    # --- 5. Map Errors to Dates and Identify Anomalies ---
    test_dates = df.index[train_size + TIME_STEPS:]
    
    min_len = min(len(test_dates), len(test_mae_loss))
    
    final_test_dates = test_dates[:min_len]
    
    # ***** THE FINAL, COMPLETE FIX *****
    # Flatten BOTH arrays to ensure they are 1-dimensional before creating the DataFrame.
    final_close_prices = df['Close'].loc[final_test_dates].values.flatten()
    final_test_mae_loss = test_mae_loss.flatten()[:min_len]

    # --- Debugging Block ---
    print("\n--- Debugging Shapes Before DataFrame Creation ---")
    print(f"final_close_prices shape:    {final_close_prices.shape}")
    print(f"final_test_mae_loss shape:   {final_test_mae_loss.shape}")
    print("------------------------------------------------\n")
    # BOTH shapes should now be (n,)

    results_df = pd.DataFrame({
        'Close': final_close_prices,
        'MAE_Loss': final_test_mae_loss
    }, index=final_test_dates)

    anomalies = results_df[results_df['MAE_Loss'] > threshold]

    print("\n\n--- Detected Anomalies ---")
    if not anomalies.empty:
        anomalies['Threshold'] = threshold
        print(anomalies)
    else:
        print("No anomalies detected with the current threshold.")

    # --- 6. Visualize the Final Results ---
    plot_anomalies(df, anomalies, TICKER)

if __name__ == '__main__':
    main()