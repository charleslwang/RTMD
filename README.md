# Stock Market Dashboard

## Overview

The Stock Market Dashboard is a comprehensive web application built with Streamlit that provides stock price analysis, predictive modeling, and news sentiment analysis. This tool helps investors make informed decisions by combining technical analysis with natural language processing of relevant news articles.

## Features

- **Real-time Stock Data**: Fetch and visualize historical stock prices
- **Advanced Forecasting Models**: 
  - ARIMA (AutoRegressive Integrated Moving Average) for statistical forecasting
  - LSTM (Long Short-Term Memory) neural networks for deep learning predictions
- **News Aggregation**: Collect recent news articles related to the selected stock
- **AI-Powered News Analysis**:
  - Automatic summarization of news articles
  - Sentiment analysis to gauge market perception
- **Interactive Dashboard**: User-friendly interface with charts, tables, and expandable content

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-market-dashboard.git
   cd stock-market-dashboard
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your News API key:
   - Register for a free API key at [News API](https://newsapi.org/)
   - Replace `"API_KEY"` in `utils/web_scraper.py` with your actual API key

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Use the sidebar to input parameters:
   - Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
   - Select a date range for analysis
   - Click "Analyze Stock" to generate insights

3. Navigate through the different tabs:
   - **ARIMA Prediction**: View statistical forecasts for the next 30 days
   - **LSTM Prediction**: Explore deep learning predictions
   - **News Summary**: Read AI-generated summaries and sentiment analysis of recent news

## Technical Details

### Stock Data Analysis
- Uses `yfinance` to fetch historical stock data
- Caches data to improve performance
- Visualizes closing prices over time

### ARIMA Model
- Implements a statistical time series model
- Parameters: (5,1,0) tuned for stock prediction
- Forecasts stock prices 30 days into the future
- Displays forecast values in both chart and table formats

### LSTM Model
- Implements a deep learning recurrent neural network
- Features two LSTM layers with 50 units each
- Trained on 80% of the available data
- Uses a 60-day window for sequence prediction
- Visualizes training, actual, and predicted values

### News Analysis
- Fetches recent news through News API
- Uses Hugging Face's transformers for:
  - Article summarization
  - Sentiment analysis (positive/negative/neutral)
- Presents news in an expandable format with confidence scores

## Project Structure

```
stock-market-dashboard/
├── app.py                  # Main Streamlit application
├── utils/
│   └── web_scraper.py      # News fetching and analysis utilities
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- yfinance
- tensorflow
- transformers
- requests
- beautifulsoup4

## Future Enhancements

- Portfolio tracking and optimization
- Integration with additional data sources
- Enhanced visualization options
- Customizable model parameters
- Backtesting capabilities for prediction models
- Real-time alerts based on price or sentiment changes

## Acknowledgments

- Data provided by Yahoo Finance
- News articles from News API
- Transformers library by Hugging Face
