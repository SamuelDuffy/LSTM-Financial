import base64
import io
import logging
import traceback

# IMPORTANT: Set Matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# --- Helper Functions (Unchanged) ---
def get_ticker_data(ticker_symbol, start_date, end_date):
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data

def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
    return np.array(sequences)

def create_lstm_autoencoder(timesteps, features):
    model = Sequential([
        LSTM(128, input_shape=(timesteps, features)),
        Dropout(rate=0.2),
        RepeatVector(timesteps),
        LSTM(128, return_sequences=True),
        Dropout(rate=0.2),
        TimeDistributed(Dense(features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_to_base64(plt_figure):
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(plt_figure)
    return img_str

# --- API Endpoint with Debugging ---
@app.route('/analyze', methods=['POST'])
def analyze_ticker():
    logging.info("Received request for /analyze")
    try:
        req_data = request.get_json()
        ticker_symbol = req_data.get('ticker')
        start_date = req_data.get('startDate')
        end_date = req_data.get('endDate')
        logging.info(f"Parameters: Ticker={ticker_symbol}, Start={start_date}, End={end_date}")

        TIME_STEPS = 30
        df_full = get_ticker_data(ticker_symbol, '2020-01-01', '2023-12-31')
        if df_full.empty:
            return jsonify({'error': f'No data found for ticker {ticker_symbol}'}), 404
        
        close_prices = df_full[['Close']].values
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(close_prices)
        sequences = create_sequences(scaled_data, TIME_STEPS)
        train_size = int(len(sequences) * 0.8)
        train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
        
        model = create_lstm_autoencoder(TIME_STEPS, features=1)
        model.fit(train_sequences, train_sequences, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
        
        train_pred = model.predict(train_sequences, verbose=0)
        test_pred = model.predict(test_sequences, verbose=0)
        train_mae_loss = np.mean(np.abs(train_pred - train_sequences), axis=1).flatten()
        test_mae_loss = np.mean(np.abs(test_pred - test_sequences), axis=1).flatten()

        threshold = np.quantile(train_mae_loss, 0.95)

        logging.info("Generating loss distribution plot...")
        fig_loss = plt.figure(figsize=(10, 5))
        sns.histplot(train_mae_loss, bins=50, kde=True, label='Train Loss', color='blue')
        sns.histplot(test_mae_loss, bins=50, kde=True, label='Test Loss', color='orange')
        plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
        loss_dist_plot_b64 = plot_to_base64(fig_loss)
        
        test_dates = df_full.index[train_size + TIME_STEPS:]
        min_len = min(len(test_dates), len(test_mae_loss))
        
        # ***** THE FINAL, CORRECTED LINE IS HERE *****
        # We must flatten the 'Close' prices array just like we flattened the MAE_Loss array.
        close_prices_for_df = df_full['Close'].iloc[train_size + TIME_STEPS : train_size + TIME_STEPS + min_len].values.flatten()
        mae_loss_for_df = test_mae_loss[:min_len]

        logging.info(f"Shapes for DataFrame: 'Close'={close_prices_for_df.shape}, 'MAE_Loss'={mae_loss_for_df.shape}")
        
        results_df = pd.DataFrame({
            'Close': close_prices_for_df,
            'MAE_Loss': mae_loss_for_df
        }, index=test_dates[:min_len])

        logging.info("Results DataFrame created successfully.")
        
        anomalies_df = results_df[results_df['MAE_Loss'] > threshold]
        df_display_range = df_full.loc[start_date:end_date]
        anomalies_in_range = anomalies_df.loc[start_date:end_date]

        logging.info("Generating anomaly plot...")
        fig_anomaly = plt.figure(figsize=(12, 6))
        plt.plot(df_display_range.index, df_display_range['Close'], label='Close Price')
        if not anomalies_in_range.empty:
            plt.scatter(anomalies_in_range.index, anomalies_in_range['Close'], color='red', label='Anomaly', s=50, zorder=5)
        anomaly_plot_b64 = plot_to_base64(fig_anomaly)

        logging.info("Analysis complete. Sending response.")
        return jsonify({
            'lossDistributionPlot': loss_dist_plot_b64,
            'anomalyPlot': anomaly_plot_b64,
            'anomalies': anomalies_in_range.reset_index().rename(columns={'index': 'date'}).to_dict(orient='records')
        })

    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"An error occurred: {e}")
        logging.error(f"Traceback:\n{error_trace}")
        return jsonify({'error': f'An internal server error occurred. Traceback: {error_trace}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)