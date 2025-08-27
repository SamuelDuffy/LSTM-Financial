import base64
import io
import logging
import traceback

# Need to set backend before importing pyplot or it crashes on headless servers
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from keras.callbacks import Callback
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class SocketIOProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        socketio.emit('epoch_update', {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss')
        })
        logging.info(f"Epoch {epoch+1} ended. Loss: {logs.get('loss')}, Val_Loss: {logs.get('val_loss')}")

def get_ticker_data(ticker_symbol, start_date, end_date):
    # yfinance automatically names the index 'Date'
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data

def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i : i + time_steps])
    return np.array(sequences)

def create_lstm_autoencoder(timesteps, features, lstm_cells=128, dropout_rate=0.2):
    model = Sequential([
        LSTM(lstm_cells, input_shape=(timesteps, features)),
        Dropout(rate=dropout_rate),
        RepeatVector(timesteps),
        LSTM(lstm_cells, return_sequences=True),
        Dropout(rate=dropout_rate),
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

@app.route('/analyze', methods=['POST'])
def analyze_ticker():
    logging.info("Received request for /analyze")
    try:
        req_data = request.get_json()
        ticker_symbol = req_data.get('ticker')
        start_date = req_data.get('startDate')
        end_date = req_data.get('endDate')
        
        time_steps = int(req_data.get('timeSteps', 30))
        lstm_cells = int(req_data.get('lstmCells', 128))
        dropout_rate = float(req_data.get('dropoutRate', 0.2))
        epochs = int(req_data.get('epochs', 10))
        
        manual_threshold = req_data.get('manualThreshold')
        if manual_threshold is not None:
            manual_threshold = float(manual_threshold)

        if not (10 <= time_steps <= 100): time_steps = 30
        if not (32 <= lstm_cells <= 256): lstm_cells = 128
        if not (0.0 <= dropout_rate <= 0.5): dropout_rate = 0.2
        if not (5 <= epochs <= 50): epochs = 10

        logging.info(f"Parameters: Ticker={ticker_symbol}, Start={start_date}, End={end_date}, "
                     f"TimeSteps={time_steps}, LstmCells={lstm_cells}, Dropout={dropout_rate}, Epochs={epochs}")

        df_full = get_ticker_data(ticker_symbol, '2019-01-01', '2024-12-31')
        if df_full.empty:
            return jsonify({'error': f'No data found for ticker {ticker_symbol}'}), 404
        
        close_prices = df_full[['Close']].values
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(close_prices)
        
        sequences = create_sequences(scaled_data, time_steps)
        if len(sequences) == 0:
            return jsonify({'error': 'Not enough data to create sequences. Try a larger date range.'}), 400
        
        train_size = int(len(sequences) * 0.8)
        train_sequences = sequences[:train_size]
        
        model = create_lstm_autoencoder(time_steps, features=1, lstm_cells=lstm_cells, dropout_rate=dropout_rate)
        history = model.fit(train_sequences, train_sequences, epochs=epochs, batch_size=32, validation_split=0.1, 
                            verbose=0, callbacks=[SocketIOProgressCallback()])
        
        all_pred = model.predict(sequences, verbose=0)
        all_mae_loss = np.mean(np.abs(all_pred - sequences), axis=1).flatten()

        train_mae_loss = all_mae_loss[:train_size]

        if manual_threshold is not None and manual_threshold > 0:
            threshold = manual_threshold
            logging.info(f"Using manually set threshold: {threshold}")
        else:
            threshold = np.quantile(train_mae_loss, 0.95)
            logging.info(f"Using 95th percentile threshold: {threshold}")

        fig_loss = plt.figure(figsize=(10, 5))
        sns.histplot(train_mae_loss, bins=50, kde=True, label='Train Loss', color='blue')
        plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
        plt.title('Reconstruction Loss Distribution (on Train Set)')
        plt.xlabel('Mean Absolute Error (MAE) Loss')
        plt.ylabel('Density')
        plt.legend()
        loss_dist_plot_b64 = plot_to_base64(fig_loss)
        
        results_dates = df_full.index[time_steps : len(all_mae_loss) + time_steps]
        
        close_prices_for_df = df_full['Close'].iloc[time_steps : len(all_mae_loss) + time_steps].values.flatten()
        
        results_df = pd.DataFrame({
            'Close': close_prices_for_df,
            'MAE_Loss': all_mae_loss
        }, index=results_dates)

        anomalies_df = results_df[results_df['MAE_Loss'] > threshold]
        
        df_display_range = df_full.loc[start_date:end_date]
        anomalies_in_range = anomalies_df.loc[start_date:end_date]

        fig_anomaly = plt.figure(figsize=(12, 6))
        plt.plot(df_display_range.index, df_display_range['Close'], label='Close Price')
        if not anomalies_in_range.empty:
            plt.scatter(anomalies_in_range.index, anomalies_in_range['Close'], color='red', label='Anomaly', s=50, zorder=5)
        plt.title(f'Price with Anomalies for {ticker_symbol}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        anomaly_plot_b64 = plot_to_base64(fig_anomaly)

        # Fixing the DataFrame column name issue - yfinance uses 'Date' but we need 'date'
        # Had to debug this for way too long because the frontend expects lowercase
        anomalies_payload = []
        if not anomalies_in_range.empty:
            # Reset index to get the Date column, then rename it to lowercase
            anomalies_payload = anomalies_in_range.reset_index().rename(columns={'Date': 'date'}).to_dict(orient='records')

            # Quick sanity check - log the first record keys to make sure rename worked
            if anomalies_payload:
                logging.info(f"Keys in the first anomaly record after rename: {list(anomalies_payload[0].keys())}")

            # Convert pandas timestamps to ISO strings for JSON serialization
            for record in anomalies_payload:
                record['date'] = record['date'].isoformat()

        logging.info("Analysis complete. Sending response.")
        return jsonify({
            'lossDistributionPlot': loss_dist_plot_b64,
            'anomalyPlot': anomaly_plot_b64,
            'anomalies': anomalies_payload,
            'modelMetrics': {
                'loss': history.history.get('loss', []),
                'val_loss': history.history.get('val_loss', [])
            },
            'thresholdUsed': threshold
        })

    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"An error occurred: {e}")
        logging.error(f"Traceback:\n{error_trace}")
        return jsonify({'error': f'An internal server error occurred. See logs for details.'}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)