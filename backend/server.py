from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import httpx
import json
from datetime import datetime, timedelta
from typing import Optional, List
import asyncio
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import motor.motor_asyncio
from pydantic import BaseModel
import uuid
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Stock Analysis Platform")

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URL = os.getenv("MONGO_URL")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client[os.getenv("DB_NAME", "stock_platform")]

# In-memory cache for demo (in production, use Redis)
cache = {}

class StockQuote(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: str
    volume: int
    timestamp: datetime

class PredictionResult(BaseModel):
    symbol: str
    model_type: str
    predictions: List[dict]
    confidence: float
    accuracy_metrics: dict

class AlphaVantageService:
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        self.use_mock = False  # Flag to use mock data when API is rate limited
    
    def get_mock_quote(self, symbol: str):
        """Return mock stock quote data"""
        mock_quotes = {
            "AAPL": {
                "Global Quote": {
                    "01. symbol": "AAPL",
                    "05. price": "175.43",
                    "09. change": "2.31",
                    "10. change percent": "1.33%",
                    "06. volume": "45678901",
                    "03. high": "177.21",
                    "04. low": "173.89",
                    "02. open": "174.12",
                    "08. previous close": "173.12",
                    "07. latest trading day": datetime.now().strftime('%Y-%m-%d')
                }
            },
            "MSFT": {
                "Global Quote": {
                    "01. symbol": "MSFT",
                    "05. price": "432.81",
                    "09. change": "-3.42",
                    "10. change percent": "-0.78%",
                    "06. volume": "32145678",
                    "03. high": "436.23",
                    "04. low": "431.45",
                    "02. open": "435.67",
                    "08. previous close": "436.23",
                    "07. latest trading day": datetime.now().strftime('%Y-%m-%d')
                }
            },
            "GOOGL": {
                "Global Quote": {
                    "01. symbol": "GOOGL",
                    "05. price": "185.92",
                    "09. change": "4.67",
                    "10. change percent": "2.58%",
                    "06. volume": "28934567",
                    "03. high": "187.34",
                    "04. low": "183.21",
                    "02. open": "184.56",
                    "08. previous close": "181.25",
                    "07. latest trading day": datetime.now().strftime('%Y-%m-%d')
                }
            },
            "TSLA": {
                "Global Quote": {
                    "01. symbol": "TSLA",
                    "05. price": "238.45",
                    "09. change": "-12.67",
                    "10. change percent": "-5.05%",
                    "06. volume": "87654321",
                    "03. high": "251.23",
                    "04. low": "236.78",
                    "02. open": "250.12",
                    "08. previous close": "251.12",
                    "07. latest trading day": datetime.now().strftime('%Y-%m-%d')
                }
            }
        }
        return mock_quotes.get(symbol.upper(), mock_quotes["AAPL"])
    
    def get_mock_historical_data(self, symbol: str):
        """Return mock historical data for ML training"""
        base_price = {"AAPL": 175, "MSFT": 433, "GOOGL": 186, "TSLA": 238}.get(symbol.upper(), 175)
        
        # Generate 100 days of mock historical data
        historical_data = {}
        for i in range(100):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            # Add some realistic price variation
            variation = np.random.normal(0, base_price * 0.02)
            price = base_price + variation
            
            historical_data[date] = {
                "1. open": str(round(price * 0.99, 2)),
                "2. high": str(round(price * 1.02, 2)),
                "3. low": str(round(price * 0.98, 2)),
                "4. close": str(round(price, 2)),
                "5. volume": str(np.random.randint(10000000, 100000000))
            }
        
        return {
            "Meta Data": {
                "1. Information": "Daily Prices (open, high, low, close) and Volumes",
                "2. Symbol": symbol.upper(),
                "3. Last Refreshed": datetime.now().strftime('%Y-%m-%d'),
                "4. Output Size": "Full size",
                "5. Time Zone": "US/Eastern"
            },
            "Time Series (Daily)": historical_data
        }
    
    def get_mock_company_overview(self, symbol: str):
        """Return mock company overview data"""
        mock_overviews = {
            "AAPL": {
                "Symbol": "AAPL",
                "Name": "Apple Inc",
                "Sector": "Technology",
                "Industry": "Consumer Electronics",
                "MarketCapitalization": "2701234567890",
                "PERatio": "28.5",
                "EPS": "6.13",
                "52WeekHigh": "199.62",
                "52WeekLow": "164.08",
                "Description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide."
            },
            "MSFT": {
                "Symbol": "MSFT",
                "Name": "Microsoft Corporation",
                "Sector": "Technology",
                "Industry": "Software—Infrastructure", 
                "MarketCapitalization": "3198765432100",
                "PERatio": "35.2",
                "EPS": "12.05",
                "52WeekHigh": "468.35",
                "52WeekLow": "362.90",
                "Description": "Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide."
            },
            "GOOGL": {
                "Symbol": "GOOGL",
                "Name": "Alphabet Inc",
                "Sector": "Technology",
                "Industry": "Internet Content & Information",
                "MarketCapitalization": "2267890123456",
                "PERatio": "22.8",
                "EPS": "8.15",
                "52WeekHigh": "193.31",
                "52WeekLow": "129.40",
                "Description": "Alphabet Inc. provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America."
            },
            "TSLA": {
                "Symbol": "TSLA",
                "Name": "Tesla Inc",
                "Sector": "Consumer Discretionary",
                "Industry": "Auto Manufacturers",
                "MarketCapitalization": "756789012345",
                "PERatio": "65.3",
                "EPS": "3.65",
                "52WeekHigh": "271.99",
                "52WeekLow": "138.80",
                "Description": "Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems in the United States, China, and internationally."
            }
        }
        return mock_overviews.get(symbol.upper(), mock_overviews["AAPL"])
    
    def get_mock_market_movers(self):
        """Return mock market movers data"""
        return {
            "metadata": "Top gainers, losers, and most actively traded US tickers",
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S US/Eastern'),
            "top_gainers": [
                {"ticker": "NVDA", "price": "145.67", "change_amount": "12.34", "change_percentage": "9.27%", "volume": "98765432"},
                {"ticker": "AMD", "price": "198.23", "change_amount": "15.67", "change_percentage": "8.59%", "volume": "76543210"},
                {"ticker": "GOOGL", "price": "185.92", "change_amount": "4.67", "change_percentage": "2.58%", "volume": "28934567"},
                {"ticker": "AAPL", "price": "175.43", "change_amount": "2.31", "change_percentage": "1.33%", "volume": "45678901"},
                {"ticker": "META", "price": "592.34", "change_amount": "7.89", "change_percentage": "1.35%", "volume": "23456789"}
            ],
            "top_losers": [
                {"ticker": "TSLA", "price": "238.45", "change_amount": "-12.67", "change_percentage": "-5.05%", "volume": "87654321"},
                {"ticker": "NFLX", "price": "712.89", "change_amount": "-28.45", "change_percentage": "-3.84%", "volume": "12345678"},
                {"ticker": "MSFT", "price": "432.81", "change_amount": "-3.42", "change_percentage": "-0.78%", "volume": "32145678"},
                {"ticker": "AMZN", "price": "178.23", "change_amount": "-2.17", "change_percentage": "-1.20%", "volume": "54321098"},
                {"ticker": "CRM", "price": "289.45", "change_amount": "-4.56", "change_percentage": "-1.55%", "volume": "19876543"}
            ],
            "most_actively_traded": [
                {"ticker": "NVDA", "price": "145.67", "change_amount": "12.34", "change_percentage": "9.27%", "volume": "198765432"},
                {"ticker": "TSLA", "price": "238.45", "change_amount": "-12.67", "change_percentage": "-5.05%", "volume": "187654321"},
                {"ticker": "AAPL", "price": "175.43", "change_amount": "2.31", "change_percentage": "1.33%", "volume": "145678901"},
                {"ticker": "AMD", "price": "198.23", "change_amount": "15.67", "change_percentage": "8.59%", "volume": "176543210"},
                {"ticker": "MSFT", "price": "432.81", "change_amount": "-3.42", "change_percentage": "-0.78%", "volume": "132145678"}
            ]
        }
    
    async def get_real_time_quote(self, symbol: str):
        cache_key = f"quote:{symbol}"
        
        # Check cache first (cache for 30 seconds)
        if cache_key in cache:
            cached_data, cached_time = cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=30):
                return cached_data
        
        # Try real API first
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.base_url, params=params)
                data = response.json()
                
                if "Error Message" in data:
                    # Symbol not found, try mock data
                    return self.get_mock_quote(symbol)
                
                if "Note" in data or "Information" in data:
                    # API rate limit exceeded, use mock data
                    print(f"API rate limited for {symbol}, using mock data")
                    return self.get_mock_quote(symbol)
                
                # Cache the result
                cache[cache_key] = (data, datetime.now())
                return data
                
        except Exception as e:
            print(f"API error for {symbol}: {e}, using mock data")
            return self.get_mock_quote(symbol)
    
    async def get_historical_data(self, symbol: str, interval: str = "daily"):
        cache_key = f"historical:{symbol}:{interval}"
        
        # Check cache first (cache for 5 minutes)
        if cache_key in cache:
            cached_data, cached_time = cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=5):
                return cached_data
        
        # Try real API first
        try:
            function_map = {
                "daily": "TIME_SERIES_DAILY",
                "weekly": "TIME_SERIES_WEEKLY",
                "monthly": "TIME_SERIES_MONTHLY"
            }
            
            params = {
                "function": function_map.get(interval, "TIME_SERIES_DAILY"),
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self.base_url, params=params)
                data = response.json()
                
                if "Error Message" in data:
                    # Symbol not found, try mock data
                    return self.get_mock_historical_data(symbol)
                
                if "Note" in data or "Information" in data:
                    # API rate limit exceeded, use mock data
                    print(f"API rate limited for historical {symbol}, using mock data")
                    return self.get_mock_historical_data(symbol)
                
                # Cache the result
                cache[cache_key] = (data, datetime.now())
                return data
                
        except Exception as e:
            print(f"API error for historical {symbol}: {e}, using mock data")
            return self.get_mock_historical_data(symbol)
    
    async def get_company_overview(self, symbol: str):
        cache_key = f"overview:{symbol}"
        
        # Check cache first (cache for 1 hour)
        if cache_key in cache:
            cached_data, cached_time = cache[cache_key]
            if datetime.now() - cached_time < timedelta(hours=1):
                return cached_data
        
        # Try real API first
        try:
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.base_url, params=params)
                data = response.json()
                
                if "Error Message" in data or not data:
                    # Symbol not found, try mock data
                    return self.get_mock_company_overview(symbol)
                
                if "Note" in data or "Information" in data:
                    # API rate limit exceeded, use mock data
                    print(f"API rate limited for overview {symbol}, using mock data")
                    return self.get_mock_company_overview(symbol)
                
                # Cache the result
                cache[cache_key] = (data, datetime.now())
                return data
                
        except Exception as e:
            print(f"API error for overview {symbol}: {e}, using mock data")
            return self.get_mock_company_overview(symbol)
    
    async def get_market_movers(self):
        cache_key = "market_movers"
        
        # Check cache first (cache for 5 minutes)
        if cache_key in cache:
            cached_data, cached_time = cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=5):
                return cached_data
        
        params = {
            "function": "TOP_GAINERS_LOSERS",
            "apikey": self.api_key
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(self.base_url, params=params)
                data = response.json()
                
                if "Note" in data:
                    raise HTTPException(status_code=429, detail="API rate limit exceeded")
                
                # Cache the result
                cache[cache_key] = (data, datetime.now())
                return data
            except httpx.TimeoutException:
                raise HTTPException(status_code=503, detail="API timeout")

class MLPredictionService:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def prepare_data(self, historical_data: dict, symbol: str):
        """Prepare historical data for ML models"""
        if "Time Series (Daily)" in historical_data:
            time_series = historical_data["Time Series (Daily)"]
        elif "Weekly Time Series" in historical_data:
            time_series = historical_data["Weekly Time Series"]
        else:
            raise ValueError("Invalid time series data format")
        
        # Convert to DataFrame
        df_data = []
        for date, values in time_series.items():
            df_data.append({
                'date': datetime.strptime(date, '%Y-%m-%d'),
                'close': float(values['4. close']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('date')
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def linear_regression_prediction(self, df: pd.DataFrame, days_ahead: int = 7):
        """Simple linear regression prediction"""
        if len(df) < 30:
            raise ValueError("Insufficient data for prediction")
        
        # Prepare features (last 30 days)
        df_recent = df.tail(60).copy()
        df_recent['day_num'] = range(len(df_recent))
        
        X = df_recent[['day_num']].values
        y = df_recent['close'].values
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future prices
        future_days = np.array([[len(df_recent) + i] for i in range(1, days_ahead + 1)])
        predictions = model.predict(future_days)
        
        # Calculate confidence (R² score)
        confidence = model.score(X, y) * 100
        
        # Generate prediction dates
        last_date = df.iloc[-1]['date']
        prediction_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
        
        return {
            "model_type": "Linear Regression",
            "predictions": [
                {"date": date.strftime('%Y-%m-%d'), "price": float(price)}
                for date, price in zip(prediction_dates, predictions)
            ],
            "confidence": float(confidence),
            "current_price": float(df.iloc[-1]['close'])
        }
    
    def prophet_prediction(self, df: pd.DataFrame, days_ahead: int = 7):
        """Prophet time series prediction"""
        try:
            # Prepare data for Prophet
            prophet_df = df[['date', 'close']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Train Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)
            
            # Extract predictions
            predictions = forecast.tail(days_ahead)[['ds', 'yhat']].to_dict('records')
            
            # Calculate confidence (simplified)
            historical_errors = abs(forecast['yhat'].iloc[:-days_ahead] - prophet_df['y'])
            confidence = max(0, 100 - (historical_errors.mean() / prophet_df['y'].mean() * 100))
            
            return {
                "model_type": "Prophet",
                "predictions": [
                    {"date": pred['ds'].strftime('%Y-%m-%d'), "price": float(pred['yhat'])}
                    for pred in predictions
                ],
                "confidence": float(confidence),
                "current_price": float(df.iloc[-1]['close'])
            }
        except Exception as e:
            # Fallback to linear regression if Prophet fails
            print(f"Prophet failed: {e}, falling back to Linear Regression")
            return self.linear_regression_prediction(df, days_ahead)

# Initialize services
alpha_service = AlphaVantageService()
ml_service = MLPredictionService()

# API Endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/stock/quote/{symbol}")
@limiter.limit("10/minute")
async def get_stock_quote(symbol: str, request: Request):
    """Get real-time stock quote"""
    try:
        data = await alpha_service.get_real_time_quote(symbol.upper())
        
        # Parse the quote data
        if "Global Quote" in data:
            quote = data["Global Quote"]
            formatted_quote = {
                "symbol": quote.get("01. symbol", symbol),
                "price": float(quote.get("05. price", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_percent": quote.get("10. change percent", "0%"),
                "volume": int(quote.get("06. volume", 0)),
                "high": float(quote.get("03. high", 0)),
                "low": float(quote.get("04. low", 0)),
                "open": float(quote.get("02. open", 0)),
                "previous_close": float(quote.get("08. previous close", 0)),
                "latest_trading_day": quote.get("07. latest trading day", "")
            }
            return {"success": True, "data": formatted_quote}
        else:
            raise HTTPException(status_code=404, detail="Quote data not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/historical/{symbol}")
@limiter.limit("5/minute")
async def get_historical_data(symbol: str, interval: str = "daily", request: Request = None):
    """Get historical stock data"""
    try:
        data = await alpha_service.get_historical_data(symbol.upper(), interval)
        return {"success": True, "data": data}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/overview/{symbol}")
@limiter.limit("5/minute")
async def get_company_overview(symbol: str, request: Request):
    """Get company fundamental data"""
    try:
        data = await alpha_service.get_company_overview(symbol.upper())
        return {"success": True, "data": data}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/movers")
@limiter.limit("3/minute")
async def get_market_movers(request: Request):
    """Get market gainers, losers, and most active"""
    try:
        data = await alpha_service.get_market_movers()
        return {"success": True, "data": data}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/{symbol}")
@limiter.limit("3/minute")
async def predict_stock_price(symbol: str, model_type: str = "linear", days_ahead: int = 7, request: Request = None):
    """Predict stock prices using ML models"""
    try:
        # Get historical data
        historical_data = await alpha_service.get_historical_data(symbol.upper(), "daily")
        
        # Prepare data for ML
        df = ml_service.prepare_data(historical_data, symbol)
        
        # Generate predictions based on model type
        if model_type.lower() == "prophet":
            prediction_result = ml_service.prophet_prediction(df, days_ahead)
        else:  # Default to linear regression
            prediction_result = ml_service.linear_regression_prediction(df, days_ahead)
        
        prediction_result["symbol"] = symbol.upper()
        prediction_result["days_ahead"] = days_ahead
        prediction_result["generated_at"] = datetime.now().isoformat()
        
        # Store prediction in database
        prediction_doc = {
            "id": str(uuid.uuid4()),
            "symbol": symbol.upper(),
            "model_type": prediction_result["model_type"],
            "predictions": prediction_result["predictions"],
            "confidence": prediction_result["confidence"],
            "generated_at": datetime.now(),
            "days_ahead": days_ahead
        }
        
        await db.predictions.insert_one(prediction_doc)
        
        return {"success": True, "data": prediction_result}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/trending")
@limiter.limit("5/minute")
async def get_trending_stocks(request: Request):
    """Get trending stocks with high volume"""
    try:
        # Popular stocks to check
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "ORCL"]
        trending_stocks = []
        
        # Get quotes for each symbol
        for symbol in symbols:
            try:
                quote_data = await alpha_service.get_real_time_quote(symbol)
                if "Global Quote" in quote_data:
                    quote = quote_data["Global Quote"]
                    trending_stocks.append({
                        "symbol": quote.get("01. symbol", symbol),
                        "price": float(quote.get("05. price", 0)),
                        "change": float(quote.get("09. change", 0)),
                        "change_percent": quote.get("10. change percent", "0%"),
                        "volume": int(quote.get("06. volume", 0))
                    })
            except:
                continue  # Skip if individual quote fails
        
        # Sort by volume (trending indicator)
        trending_stocks.sort(key=lambda x: x["volume"], reverse=True)
        
        return {"success": True, "data": trending_stocks[:8]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/volatile")
@limiter.limit("5/minute") 
async def get_volatile_stocks(request: Request):
    """Get volatile stocks with high price swings"""
    try:
        # Get market movers data
        movers_data = await alpha_service.get_market_movers()
        
        volatile_stocks = []
        
        # Extract gainers and losers as volatile stocks
        if "top_gainers" in movers_data:
            for stock in movers_data["top_gainers"][:5]:
                volatile_stocks.append({
                    "symbol": stock["ticker"],
                    "price": float(stock["price"]),
                    "change_amount": float(stock["change_amount"]),
                    "change_percentage": stock["change_percentage"],
                    "volume": int(stock["volume"]),
                    "type": "gainer"
                })
        
        if "top_losers" in movers_data:
            for stock in movers_data["top_losers"][:5]:
                volatile_stocks.append({
                    "symbol": stock["ticker"],
                    "price": float(stock["price"]),
                    "change_amount": float(stock["change_amount"]),
                    "change_percentage": stock["change_percentage"],
                    "volume": int(stock["volume"]),
                    "type": "loser"
                })
        
        return {"success": True, "data": volatile_stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
