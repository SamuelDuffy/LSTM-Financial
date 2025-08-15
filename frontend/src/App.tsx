import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

interface Anomaly {
    date: string;
    Close: number;
    MAE_Loss: number;
}

interface AnalysisResults {
    lossDistributionPlot: string;
    anomalyPlot: string;
    anomalies: Anomaly[];
}

const App: React.FC = () => {
    const [ticker, setTicker] = useState<string>('TSLA');
    const [startDate, setStartDate] = useState<string>('2022-01-01');
    const [endDate, setEndDate] = useState<string>('2023-12-31');
    const [results, setResults] = useState<AnalysisResults | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');

    const handleAnalyze = async () => {
        setLoading(true);
        setError('');
        setResults(null);
        try {
            const response = await axios.post<AnalysisResults>('http://127.0.0.1:5000/analyze', {
                ticker,
                startDate,
                endDate
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

    const handleTickerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setTicker(e.target.value.toUpperCase());
    };

    const handleStartDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setStartDate(e.target.value);
    };

    const handleEndDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setEndDate(e.target.value);
    };

    return (
        <div className="container">
            <header className="site-header">
                <h1>Stock Price Anomaly Detector</h1>
                <p className="subtitle">
                    Enter a ticker and a date range to find anomalies. The model is trained on historical data from 2020-2023.
                </p>
            </header>

            <main>
                <div className="controls" role="region" aria-label="controls">
                    <div className="input-group">
                        <label>Ticker Symbol</label>
                        <input
                            type="text"
                            value={ticker}
                            onChange={handleTickerChange}
                            placeholder="e.g., AAPL"
                        />
                    </div>

                    <div className="input-group">
                        <label>Start Date</label>
                        <input type="date" value={startDate} onChange={handleStartDateChange} />
                    </div>

                    <div className="input-group">
                        <label>End Date</label>
                        <input type="date" value={endDate} onChange={handleEndDateChange} />
                    </div>

                    <button onClick={handleAnalyze} disabled={loading}>
                        {loading ? 'Analyzing...' : 'Analyze'}
                    </button>
                </div>

                {loading && <div className="loader" aria-hidden="true"></div>}

                {error && <p className="error-message">Error: {error}</p>}

                {results && (
                    <div className="results-container">
                        <h2>Analysis Results for {ticker}</h2>

                        <div className="plots">
                            <div className="plot-card">
                                <h3>Price with Anomalies</h3>
                                <img src={`data:image/png;base64,${results.anomalyPlot}`} alt="Anomaly Plot" />
                            </div>

                            <div className="plot-card">
                                <h3>Reconstruction Loss Distribution</h3>
                                <img src={`data:image/png;base64,${results.lossDistributionPlot}`} alt="Loss Distribution" />
                            </div>
                        </div>

                        <div className="anomalies-table">
                            <h3>Detected Anomalies in Range</h3>
                            {results.anomalies.length > 0 ? (
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Date</th>
                                            <th>Close Price</th>
                                            <th>MAE Loss</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {results.anomalies.map((anomaly, index) => (
                                            <tr key={index}>
                                                <td>{new Date(anomaly.date).toLocaleDateString()}</td>
                                                <td>${anomaly.Close.toFixed(2)}</td>
                                                <td>{anomaly.MAE_Loss.toFixed(4)}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            ) : (
                                <p>No anomalies detected in the selected date range.</p>
                            )}
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default App;
