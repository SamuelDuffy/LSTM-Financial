# Stock Price Anomaly Detector

A full-stack application that uses LSTM (Long Short-Term Memory) autoencoders to detect anomalies in stock price data. The system learns normal price patterns and flags unusual price movements that deviate significantly from expected behavior.

## Features

- **Real-time Training Updates**: WebSocket connection provides live training progress
- **Interactive Visualizations**: Chart.js integration for loss curves and matplotlib-generated anomaly plots
- **Customizable Parameters**: Adjustable LSTM hyperparameters and detection thresholds
- **Historical Analysis**: Analyze any date range with automatic data fetching from Yahoo Finance
- **Anomaly Detection**: Automatically identifies unusual price movements using reconstruction error

## Architecture

### Frontend (React + TypeScript)
- Modern React application with TypeScript for type safety
- Real-time WebSocket updates during model training
- Interactive charts using Chart.js and react-chartjs-2
- Responsive design with custom CSS

### Backend (Flask + Python)
- Flask API with Socket.IO for real-time communication
- TensorFlow/Keras for LSTM autoencoder implementation
- Yahoo Finance integration for stock data retrieval
- Base64-encoded plot generation for frontend display

## Prerequisites

### Backend Dependencies
- Python 3.8+
- TensorFlow/Keras
- Flask & Flask-SocketIO
- yfinance
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

### Frontend Dependencies
- Node.js 14+
- React 18+
- TypeScript
- Chart.js
- Axios
- Socket.IO client

## Installation

### Backend Setup

1. Clone the repository and navigate to the backend directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install flask flask-socketio flask-cors tensorflow pandas numpy matplotlib seaborn yfinance scikit-learn
   ```

4. Run the Flask server:
   ```bash
   python app.py
   ```
   Server will start on `http://127.0.0.1:5000`

### Frontend Setup

1. Navigate to the frontend directory
2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   Application will open at `http://localhost:3000`

## Usage

### Basic Analysis

1. **Set Parameters**: Choose a stock ticker (e.g., TSLA, AAPL) and date range
2. **Configure Model**: Adjust LSTM hyperparameters if desired:
   - Time Steps (10-100): How many previous days to consider
   - LSTM Cells (32-256): Model complexity
   - Dropout (0.0-0.5): Regularization to prevent overfitting
   - Epochs (5-50): Training iterations

3. **Run Analysis**: Click "Analyze" to start the process
4. **Monitor Training**: Watch real-time epoch updates via WebSocket
5. **Review Results**: Examine detected anomalies and model performance

### Understanding Results

- **Loss Distribution**: Shows the reconstruction error distribution with threshold line
- **Price with Anomalies**: Time series plot highlighting detected anomalies in red
- **Anomaly Table**: Detailed list of flagged dates with prices and error values
- **Model Metrics**: Training and validation loss curves

## How It Works

### LSTM Autoencoder Approach

1. **Data Preprocessing**: Stock prices are standardized using StandardScaler
2. **Sequence Creation**: Creates sliding windows of historical prices
3. **Model Architecture**:
   - Encoder LSTM compresses price sequences into fixed-size representation
   - Decoder LSTM reconstructs the original sequence
   - Higher reconstruction error indicates anomalous behavior

4. **Threshold Selection**: Uses 95th percentile of training reconstruction errors (or manual threshold)
5. **Anomaly Detection**: Points exceeding threshold are flagged as anomalies

### Model Architecture
```
Input (time_steps, 1) 
    ↓
LSTM Encoder (128 cells) + Dropout
    ↓
RepeatVector (reconstruction preparation)
    ↓
LSTM Decoder (128 cells) + Dropout
    ↓
TimeDistributed Dense (output reconstruction)
```

## Configuration Options

### Model Hyperparameters

- **Time Steps**: Number of previous days used for prediction (default: 30)
- **LSTM Cells**: Hidden units in LSTM layers (default: 128)
- **Dropout Rate**: Regularization strength (default: 0.2)
- **Epochs**: Training iterations (default: 10)

### Detection Settings

- **Manual Threshold**: Override automatic threshold calculation
- **Date Range**: Specify analysis period
- **Ticker Symbol**: Any valid stock symbol from Yahoo Finance

## API Endpoints

### POST /analyze
Analyzes stock data for anomalies.

**Request Body:**
```json
{
  "ticker": "TSLA",
  "startDate": "2022-01-01",
  "endDate": "2023-12-31",
  "timeSteps": 30,
  "lstmCells": 128,
  "dropoutRate": 0.2,
  "epochs": 10,
  "manualThreshold": null
}
```

**Response:**
```json
{
  "lossDistributionPlot": "base64_encoded_image",
  "anomalyPlot": "base64_encoded_image",
  "anomalies": [
    {
      "date": "2022-03-15T00:00:00",
      "Close": 850.25,
      "MAE_Loss": 0.0234
    }
  ],
  "modelMetrics": {
    "loss": [0.1, 0.08, 0.06],
    "val_loss": [0.12, 0.09, 0.07]
  },
  "thresholdUsed": 0.0189
}
```

## WebSocket Events

- **Connection**: `connect` - Establishes real-time communication
- **Training Updates**: `epoch_update` - Sends loss metrics during training

## Troubleshooting

### Common Issues

1. **"No data found for ticker"**: Verify ticker symbol and date range
2. **"Not enough data to create sequences"**: Extend date range or reduce time steps
3. **Backend connection errors**: Ensure Flask server is running on port 5000
4. **WebSocket connection failed**: Check CORS settings and firewall

### Performance Considerations

- Larger datasets require more memory and training time
- Higher LSTM cell counts increase model complexity but may improve accuracy
- More epochs generally improve performance but increase training time

## Technical Notes

- Uses headless matplotlib backend (`Agg`) for server environments
- Automatic data validation and parameter bounds checking
- Error handling with detailed logging for debugging
- Base64 encoding for seamless image transfer to frontend

## Future Enhancements

- Support for multiple stocks simultaneously
- Additional anomaly detection algorithms
- Model persistence and retraining capabilities
- Advanced visualization options
- Email/SMS alerts for detected anomalies

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

*This is a certified Sam project* 🚀
