import requests
import sys
import time
from datetime import datetime

class StockAPITester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=15)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, params=params, timeout=15)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                return success, response.json()
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    print(f"Response: {response.json()}")
                except:
                    print(f"Response: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health check endpoint"""
        return self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )

    def test_stock_quote(self, symbol):
        """Test stock quote endpoint"""
        return self.run_test(
            f"Stock Quote for {symbol}",
            "GET",
            f"api/stock/quote/{symbol}",
            200
        )

    def test_historical_data(self, symbol, interval="daily"):
        """Test historical data endpoint"""
        return self.run_test(
            f"Historical Data for {symbol} ({interval})",
            "GET",
            f"api/stock/historical/{symbol}",
            200,
            params={"interval": interval}
        )

    def test_company_overview(self, symbol):
        """Test company overview endpoint"""
        return self.run_test(
            f"Company Overview for {symbol}",
            "GET",
            f"api/stock/overview/{symbol}",
            200
        )

    def test_market_movers(self):
        """Test market movers endpoint"""
        return self.run_test(
            "Market Movers",
            "GET",
            "api/market/movers",
            200
        )

    def test_trending_stocks(self):
        """Test trending stocks endpoint"""
        return self.run_test(
            "Trending Stocks",
            "GET",
            "api/market/trending",
            200
        )

    def test_volatile_stocks(self):
        """Test volatile stocks endpoint"""
        return self.run_test(
            "Volatile Stocks",
            "GET",
            "api/market/volatile",
            200
        )

    def test_ai_prediction(self, symbol, model_type="linear", days_ahead=7):
        """Test AI prediction endpoint"""
        return self.run_test(
            f"AI Prediction for {symbol} using {model_type} model for {days_ahead} days",
            "POST",
            f"api/predict/{symbol}",
            200,
            params={"model_type": model_type, "days_ahead": days_ahead}
        )

    def run_all_tests(self):
        """Run all API tests"""
        print(f"🚀 Starting API tests against {self.base_url}")
        
        # Test health check
        self.test_health_check()
        
        # Test with multiple symbols
        for symbol in self.test_symbols:
            # Basic rate limiting to avoid API throttling
            time.sleep(1)
            
            # Test stock quote
            success, quote_data = self.test_stock_quote(symbol)
            
            if success:
                # Test company overview
                time.sleep(1)
                self.test_company_overview(symbol)
                
                # Test historical data
                time.sleep(1)
                self.test_historical_data(symbol)
                
                # Test AI prediction with linear model
                time.sleep(1)
                self.test_ai_prediction(symbol, "linear", 7)
                
                # Test AI prediction with prophet model
                time.sleep(1)
                self.test_ai_prediction(symbol, "prophet", 7)
        
        # Test market data endpoints
        time.sleep(1)
        self.test_market_movers()
        
        time.sleep(1)
        self.test_trending_stocks()
        
        time.sleep(1)
        self.test_volatile_stocks()
        
        # Print results
        print(f"\n📊 Tests passed: {self.tests_passed}/{self.tests_run} ({(self.tests_passed/self.tests_run)*100:.1f}%)")
        
        return self.tests_passed == self.tests_run

def main():
    # Get the backend URL from the frontend .env file
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from frontend .env
    load_dotenv('/app/frontend/.env')
    
    # Get the backend URL
    backend_url = os.getenv('REACT_APP_BACKEND_URL')
    
    if not backend_url:
        print("❌ Error: REACT_APP_BACKEND_URL not found in frontend/.env")
        return 1
    
    print(f"Using backend URL: {backend_url}")
    
    # Setup tester
    tester = StockAPITester(backend_url)
    
    # Run all tests
    success = tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
