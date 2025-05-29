import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [symbol, setSymbol] = useState('AAPL');
  const [quote, setQuote] = useState(null);
  const [overview, setOverview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [marketMovers, setMarketMovers] = useState(null);
  const [trendingStocks, setTrendingStocks] = useState(null);
  const [volatileStocks, setVolatileStocks] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionModel, setPredictionModel] = useState('linear');
  const [predictionDays, setPredictionDays] = useState(7);

  // Fetch data functions
  const fetchWithErrorHandling = async (url) => {
    try {
      const response = await fetch(url);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'API request failed');
      }
      
      return data;
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  };

  const fetchStockQuote = async (stockSymbol) => {
    try {
      const data = await fetchWithErrorHandling(`${API_BASE_URL}/api/stock/quote/${stockSymbol}`);
      return data.data;
    } catch (error) {
      throw error;
    }
  };

  const fetchCompanyOverview = async (stockSymbol) => {
    try {
      const data = await fetchWithErrorHandling(`${API_BASE_URL}/api/stock/overview/${stockSymbol}`);
      return data.data;
    } catch (error) {
      throw error;
    }
  };

  const fetchPrediction = async (stockSymbol, modelType, days) => {
    try {
      const data = await fetchWithErrorHandling(`${API_BASE_URL}/api/predict/${stockSymbol}?model_type=${modelType}&days_ahead=${days}`);
      return data.data;
    } catch (error) {
      throw error;
    }
  };

  const fetchMarketMovers = async () => {
    try {
      const data = await fetchWithErrorHandling(`${API_BASE_URL}/api/market/movers`);
      return data.data;
    } catch (error) {
      throw error;
    }
  };

  const fetchTrendingStocks = async () => {
    try {
      const data = await fetchWithErrorHandling(`${API_BASE_URL}/api/market/trending`);
      return data.data;
    } catch (error) {
      throw error;
    }
  };

  const fetchVolatileStocks = async () => {
    try {
      const data = await fetchWithErrorHandling(`${API_BASE_URL}/api/market/volatile`);
      return data.data;
    } catch (error) {
      throw error;
    }
  };

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const [quoteData, overviewData, trendingData, volatileData, moversData] = await Promise.allSettled([
          fetchStockQuote(symbol),
          fetchCompanyOverview(symbol),
          fetchTrendingStocks(),
          fetchVolatileStocks(),
          fetchMarketMovers()
        ]);

        if (quoteData.status === 'fulfilled') setQuote(quoteData.value);
        if (overviewData.status === 'fulfilled') setOverview(overviewData.value);
        if (trendingData.status === 'fulfilled') setTrendingStocks(trendingData.value);
        if (volatileData.status === 'fulfilled') setVolatileStocks(volatileData.value);
        if (moversData.status === 'fulfilled') setMarketMovers(moversData.value);
        
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, []);

  // Handle symbol search
  const handleSymbolSubmit = async (e) => {
    e.preventDefault();
    const newSymbol = e.target.symbol.value.toUpperCase();
    setSymbol(newSymbol);
    
    setLoading(true);
    setError(null);
    
    try {
      const [quoteData, overviewData] = await Promise.all([
        fetchStockQuote(newSymbol),
        fetchCompanyOverview(newSymbol)
      ]);
      
      setQuote(quoteData);
      setOverview(overviewData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle AI prediction
  const handlePrediction = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const predictionData = await fetchPrediction(symbol, predictionModel, predictionDays);
      setPrediction(predictionData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Format number with commas
  const formatNumber = (num) => {
    return new Intl.NumberFormat().format(num);
  };

  // Format currency
  const formatCurrency = (num) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(num);
  };

  const renderDashboard = () => (
    <div className="space-y-6">
      {/* Market Overview */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">🏛️ Market Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-800">Market Status</h3>
            <p className="text-blue-600">Real-time Data Active</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-green-800">AI Models</h3>
            <p className="text-green-600">Linear Regression & Prophet</p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-purple-800">Data Source</h3>
            <p className="text-purple-600">Alpha Vantage API</p>
          </div>
        </div>
      </div>

      {/* Stock Search */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">🔍 Stock Analysis</h2>
        <form onSubmit={handleSymbolSubmit} className="mb-6">
          <div className="flex gap-3">
            <input
              type="text"
              name="symbol"
              placeholder="Enter stock symbol (e.g., AAPL, MSFT)"
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              defaultValue={symbol}
            />
            <button
              type="submit"
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              disabled={loading}
            >
              {loading ? 'Loading...' : 'Analyze'}
            </button>
          </div>
        </form>

        {/* Real-time Quote */}
        {quote && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-xl font-semibold mb-3 text-gray-800">📈 Real-time Quote</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Symbol:</span>
                  <span className="font-semibold">{quote.symbol}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Price:</span>
                  <span className="font-semibold text-blue-600">{formatCurrency(quote.price)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Change:</span>
                  <span className={`font-semibold ${quote.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {quote.change >= 0 ? '+' : ''}{quote.change} ({quote.change_percent})
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Volume:</span>
                  <span className="font-semibold">{formatNumber(quote.volume)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">High:</span>
                  <span className="font-semibold">{formatCurrency(quote.high)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Low:</span>
                  <span className="font-semibold">{formatCurrency(quote.low)}</span>
                </div>
              </div>
            </div>

            {/* Company Overview */}
            {overview && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-xl font-semibold mb-3 text-gray-800">🏢 Company Overview</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Name:</span>
                    <span className="font-semibold text-right">{overview.Name || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Sector:</span>
                    <span className="font-semibold">{overview.Sector || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Market Cap:</span>
                    <span className="font-semibold">{overview.MarketCapitalization ? `$${formatNumber(overview.MarketCapitalization)}` : 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">P/E Ratio:</span>
                    <span className="font-semibold">{overview.PERatio || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">EPS:</span>
                    <span className="font-semibold">{overview.EPS || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">52W High:</span>
                    <span className="font-semibold">{overview['52WeekHigh'] ? formatCurrency(overview['52WeekHigh']) : 'N/A'}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );

  const renderPrediction = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">🤖 AI Price Prediction</h2>
        
        {/* Prediction Controls */}
        <div className="bg-gray-50 p-4 rounded-lg mb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Stock Symbol</label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">AI Model</label>
              <select
                value={predictionModel}
                onChange={(e) => setPredictionModel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="linear">Linear Regression</option>
                <option value="prophet">Prophet (Time Series)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Prediction Days</label>
              <select
                value={predictionDays}
                onChange={(e) => setPredictionDays(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value={1}>1 Day</option>
                <option value={3}>3 Days</option>
                <option value={7}>7 Days</option>
                <option value={14}>14 Days</option>
                <option value={30}>30 Days</option>
              </select>
            </div>
          </div>
          <button
            onClick={handlePrediction}
            disabled={loading}
            className="mt-4 px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:bg-gray-400"
          >
            {loading ? 'Generating Prediction...' : 'Generate AI Prediction'}
          </button>
        </div>

        {/* Prediction Results */}
        {prediction && (
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold mb-4 text-gray-800">Prediction Results</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">{prediction.model_type}</div>
                  <div className="text-sm text-gray-600">AI Model</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{prediction.confidence.toFixed(1)}%</div>
                  <div className="text-sm text-gray-600">Confidence Score</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{formatCurrency(prediction.current_price)}</div>
                  <div className="text-sm text-gray-600">Current Price</div>
                </div>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="text-lg font-semibold mb-3 text-gray-800">📊 Price Predictions</h4>
              <div className="space-y-2">
                {prediction.predictions.map((pred, index) => {
                  const priceChange = pred.price - prediction.current_price;
                  const priceChangePercent = (priceChange / prediction.current_price) * 100;
                  return (
                    <div key={index} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-b-0">
                      <span className="text-gray-600">{pred.date}</span>
                      <div className="text-right">
                        <span className="font-semibold">{formatCurrency(pred.price)}</span>
                        <span className={`ml-2 text-sm ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          ({priceChange >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%)
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderMarketData = () => (
    <div className="space-y-6">
      {/* Trending Stocks */}
      {trendingStocks && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">🔥 Trending Stocks</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {trendingStocks.map((stock, index) => (
              <div key={index} className="bg-blue-50 p-4 rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <span className="font-bold text-lg">{stock.symbol}</span>
                  <span className="text-sm text-gray-600">#{index + 1}</span>
                </div>
                <div className="space-y-1">
                  <div className="text-xl font-semibold text-blue-600">{formatCurrency(stock.price)}</div>
                  <div className={`text-sm font-medium ${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {stock.change >= 0 ? '+' : ''}{stock.change} ({stock.change_percent})
                  </div>
                  <div className="text-xs text-gray-500">Volume: {formatNumber(stock.volume)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Volatile Stocks */}
      {volatileStocks && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">⚡ Volatile Stocks</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-3 text-green-600">🚀 Top Gainers</h3>
              <div className="space-y-2">
                {volatileStocks.filter(stock => stock.type === 'gainer').map((stock, index) => (
                  <div key={index} className="flex justify-between items-center bg-green-50 p-3 rounded-lg">
                    <div>
                      <span className="font-semibold">{stock.symbol}</span>
                      <div className="text-sm text-gray-600">{formatCurrency(stock.price)}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-green-600 font-semibold">{stock.change_percentage}</div>
                      <div className="text-xs text-gray-500">{formatNumber(stock.volume)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-3 text-red-600">📉 Top Losers</h3>
              <div className="space-y-2">
                {volatileStocks.filter(stock => stock.type === 'loser').map((stock, index) => (
                  <div key={index} className="flex justify-between items-center bg-red-50 p-3 rounded-lg">
                    <div>
                      <span className="font-semibold">{stock.symbol}</span>
                      <div className="text-sm text-gray-600">{formatCurrency(stock.price)}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-red-600 font-semibold">{stock.change_percentage}</div>
                      <div className="text-xs text-gray-500">{formatNumber(stock.volume)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Market Movers */}
      {marketMovers && marketMovers.top_gainers && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">📊 Market Movers</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-3 text-green-600">Top Gainers</h3>
              <div className="space-y-2">
                {marketMovers.top_gainers.slice(0, 5).map((stock, index) => (
                  <div key={index} className="flex justify-between items-center p-2 hover:bg-gray-50 rounded">
                    <span className="font-medium">{stock.ticker}</span>
                    <span className="text-green-600 font-semibold">{stock.change_percentage}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-3 text-red-600">Top Losers</h3>
              <div className="space-y-2">
                {marketMovers.top_losers.slice(0, 5).map((stock, index) => (
                  <div key={index} className="flex justify-between items-center p-2 hover:bg-gray-50 rounded">
                    <span className="font-medium">{stock.ticker}</span>
                    <span className="text-red-600 font-semibold">{stock.change_percentage}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return renderDashboard();
      case 'prediction':
        return renderPrediction();
      case 'market':
        return renderMarketData();
      default:
        return renderDashboard();
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">📈 AI Stock Analysis Platform</h1>
            </div>
            <div className="flex space-x-1">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-2 rounded-md transition-colors ${
                  activeTab === 'dashboard'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('prediction')}
                className={`px-4 py-2 rounded-md transition-colors ${
                  activeTab === 'prediction'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                AI Prediction
              </button>
              <button
                onClick={() => setActiveTab('market')}
                className={`px-4 py-2 rounded-md transition-colors ${
                  activeTab === 'market'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                Market Data
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            <strong>Error:</strong> {error}
          </div>
        )}
        
        {loading && (
          <div className="mb-6 bg-blue-50 border border-blue-200 text-blue-700 px-4 py-3 rounded-lg">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-700 mr-2"></div>
              Loading data...
            </div>
          </div>
        )}

        {renderContent()}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="text-center text-gray-600">
            <p>AI-Powered Stock Analysis Platform • Real-time Data • Machine Learning Predictions</p>
            <p className="text-sm mt-1">Data provided by Alpha Vantage • Predictions are for educational purposes only</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
