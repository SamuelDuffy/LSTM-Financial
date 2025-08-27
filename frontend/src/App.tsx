import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { io, Socket } from 'socket.io-client';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import './App.css';

// Dropped the annotation plugin - wasn't using it anyway
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface Anomaly {
    date: string;
    Close: number;
    MAE_Loss: number;
}

interface ModelMetrics {
    loss: number[];
    val_loss: number[];
}

// Cleaned up the interface - removed unused fullData prop
interface AnalysisResults {
    lossDistributionPlot: string;
    anomalyPlot: string;
    anomalies: Anomaly[];
    modelMetrics: ModelMetrics;
    thresholdUsed: number;
}

interface EpochUpdate {
    epoch: number;
    loss: number;
    val_loss: number;
}

const App: React.FC = () => {
    // Basic inputs
    const [ticker, setTicker] = useState<string>('TSLA');
    const [startDate, setStartDate] = useState<string>('2022-01-01');
    const [endDate, setEndDate] = useState<string>('2023-12-31');
    
    // Model settings
    const [timeSteps, setTimeSteps] = useState<number>(30);
    const [lstmCells, setLstmCells] = useState<number>(128);
    const [dropoutRate, setDropoutRate] = useState<number>(0.2);
    const [epochs, setEpochs] = useState<number>(10);
    const [manualThreshold, setManualThreshold] = useState<string>('');

    // App state
    const [results, setResults] = useState<AnalysisResults | null>(null);
    const [trainingLog, setTrainingLog] = useState<EpochUpdate[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');

    const socketRef = useRef<Socket | null>(null);

    useEffect(() => {
        socketRef.current = io('http://127.0.0.1:5000');
        socketRef.current.on('connect', () => console.log('Connected to WebSocket server'));
        socketRef.current.on('epoch_update', (data: EpochUpdate) => {
            setTrainingLog(prevLog => [...prevLog, data]);
        });
        return () => {
            if (socketRef.current) socketRef.current.disconnect();
        };
    }, []);

    const handleAnalyze = async () => {
        setLoading(true);
        setError('');
        setResults(null);
        setTrainingLog([]);
        try {
            const response = await axios.post<AnalysisResults>('http://127.0.0.1:5000/analyze', {
                ticker,
                startDate,
                endDate,
                timeSteps,
                lstmCells,
                dropoutRate,
                epochs,
                manualThreshold: manualThreshold ? parseFloat(manualThreshold) : null
            });

            if ((response.data as any).error) {
                setError((response.data as any).error);
            } else {
                setResults(response.data);
            }
        } catch (err: any) {
            const errorMessage = err.response?.data?.error || 'An unexpected error occurred.';
            setError(errorMessage);
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const lossChartData = {
        labels: results?.modelMetrics.loss.map((_, index) => `Epoch ${index + 1}`),
        datasets: [
            { label: 'Training Loss', data: results?.modelMetrics.loss, borderColor: 'rgb(54, 162, 235)', yAxisID: 'y', tension: 0.1 },
            { label: 'Validation Loss', data: results?.modelMetrics.val_loss, borderColor: 'rgb(255, 99, 132)', yAxisID: 'y', tension: 0.1 },
        ],
    };

    // Removed anomaly plot chart data - using backend generated image instead

    return (
        <div className="container">
            <header className="site-header">
                <h1>Stock Price Anomaly Detector</h1>
            </header>

            <main>

                <div className="section-card">
                    <h2>Introduction</h2>
                    <p className="subtitle">
                       LSTMs are a useful way to spot unusual patterns in time-series data. This site shows how they can be applied to stock prices to highlight potential anomalies. 
                    </p>
                </div>

                <div className="section-card">
                    <h2>Analysis Parameters</h2>
                    <p className="section-description">
                        Choose a stock and the date range you'd like to analyze.
                    </p>
                    <div className="controls-grid" role="region" aria-label="Analysis Input Controls">
                        <div className="input-group"><label>Ticker Symbol</label><input type="text" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} placeholder="e.g., AAPL"/></div>
                        <div className="input-group"><label>Start Date</label><input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} /></div>
                        <div className="input-group"><label>End Date</label><input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} /></div>
                    </div>
                </div>

                <div className="section-card">
                    <h2>Model Hyperparameters</h2>
                    <p className="section-description">
                        Adjust the settings that shape how the LSTM model learns. Try different values to see how they affect the results.
                    </p>
                    <div className="controls-grid" role="region" aria-label="Model Hyperparameters">
                        <div className="input-group"><label>Time Steps (10-100)</label><input type="number" value={timeSteps} onChange={(e) => setTimeSteps(parseInt(e.target.value))} /></div>
                        <div className="input-group"><label>LSTM Cells (32-256)</label><input type="number" value={lstmCells} onChange={(e) => setLstmCells(parseInt(e.target.value))} /></div>
                        <div className="input-group"><label>Dropout (0.0-0.5)</label><input type="number" step="0.1" value={dropoutRate} onChange={(e) => setDropoutRate(parseFloat(e.target.value))} /></div>
                        <div className="input-group"><label>Epochs (5-50)</label><input type="number" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} /></div>
                        <div className="input-group">
                            <label>Manual Threshold (Optional)</label>
                            <input type="number" step="0.001" value={manualThreshold} onChange={(e) => setManualThreshold(e.target.value)} placeholder="Default: 95th Percentile" />
                        </div>
                    </div>
                </div>
                
                <div className="button-container">
                    <button onClick={handleAnalyze} disabled={loading}>{loading ? 'Analyzing...' : 'Analyze'}</button>
                </div>

                {loading && (
                    <div className="training-status section-card">
                        <h3>Training in Progress...</h3><div className="loader"></div>
                        <div className="training-log">
                            <h4>Epoch Log:</h4>
                            <ul>{trainingLog.map(log => (<li key={log.epoch}>Epoch {log.epoch}: Loss - {log.loss?.toFixed(6)}, Val Loss - {log.val_loss?.toFixed(6)}</li>))}</ul>
                        </div>
                    </div>
                )}

                {error && <p className="error-message section-card">Error: {error}</p>}

                {results && (
                    <div className="results-container">
                        <h2 className="results-title">Analysis Results for {ticker}</h2>
                        <p className="results-summary section-card">
                            <strong>Threshold Used for Anomalies:</strong> {results.thresholdUsed.toFixed(4)} <br/>
                            <small>(Anything above this reconstruction error is flagged as an anomaly)</small>
                        </p>
                        
                        <div className="plots-section-top">
                            <div className="plot-card"><h3>Model Loss History</h3><Line options={{ responsive: true, plugins: { legend: { position: 'top' }, title: { display: true, text: 'Training & Validation Loss' }}}} data={lossChartData} /></div>
                            <div className="plot-card"><h3>Reconstruction Loss Distribution</h3><img src={`data:image/png;base64,${results.lossDistributionPlot}`} alt="Reconstruction Loss Distribution" /></div>
                        </div>

                        {/* Just using a simple img tag here - the backend handles all the plotting */}
                        <div className="plot-card full-width">
                            <h3>Price with Anomalies</h3>
                            <img src={`data:image/png;base64,${results.anomalyPlot}`} alt="Anomaly Plot" />
                        </div>

                        <div className="anomalies-table section-card">
                            <h3>Detected Anomalies in Range</h3>
                            {results.anomalies.length > 0 ? (
                                <table>
                                    <thead><tr><th>Date</th><th>Close Price</th><th>MAE Loss</th></tr></thead>
                                    <tbody>
                                        {results.anomalies.map((anomaly, index) => (
                                            <tr key={index}>
                                                <td>{new Date(anomaly.date).toLocaleString()}</td>
                                                <td>${anomaly.Close.toFixed(2)}</td>
                                                <td>{anomaly.MAE_Loss.toFixed(4)}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            ) : (<p>No anomalies detected in the selected date range. Maybe try adjusting the threshold or date range?</p>)}
                        </div>
                    </div>
                )}
            </main>
            <footer className="site-footer">
                <p>This is a certified Sam project</p>
            </footer>
        </div>
    );
};

export default App;