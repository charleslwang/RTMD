Stock Market Dashboard with ARIMA and LSTM Predictions
This project is a Stock Market Dashboard built with Streamlit, providing insights into stock price predictions and recent company performance. The app uses ARIMA and LSTM models to forecast stock prices and employs NLP to generate AI-driven summaries and sentiment analysis from recent news articles.

Features
Stock Price Predictions:
ARIMA Model for short-term linear forecasting.
LSTM Model for capturing complex patterns and non-linear trends.
Recent News Summary and Sentiment Analysis:
Scrapes the latest news about a company and generates a summary.
Performs sentiment analysis to gauge public perception and predict potential stock behavior.
Demo
Run the app locally to explore features like stock predictions, news summaries, and sentiment analysis. The dashboard provides interactive controls for selecting stocks, date ranges, and viewing multiple analyses in tabs.

Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/stock-market-dashboard.git
cd stock-market-dashboard
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Usage
Enter Stock Symbol: Type in the stock ticker (e.g., AAPL for Apple).
Select Date Range: Choose the start and end dates for analysis.
Analyze Stock:
The app displays an overview of the stock's historical prices.
ARIMA Prediction and LSTM Prediction tabs provide forecast charts and predicted values.
News Summary tab displays AI-generated summaries of recent news and sentiment analysis.
Project Structure
bash
Copy code
.
├── app.py               # Main Streamlit app code
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
Requirements
Python 3.7+
Install dependencies with pip install -r requirements.txt
Dependencies
streamlit: For creating the interactive dashboard
pandas, numpy: Data manipulation
matplotlib: Data visualization
statsmodels: ARIMA model for time series forecasting
tensorflow: LSTM model for deep learning
yfinance: Fetch stock data from Yahoo Finance
requests, beautifulsoup4: Web scraping for recent news
transformers: NLP model for text summarization and sentiment analysis
Notes
Rate Limits: Web scraping may be subject to rate limits. Consider using an API like the News API for higher reliability.
Expandability: This dashboard can be expanded to include additional analysis or data sources as needed.
