import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from web_scraper import get_news, summarize_news, analyze_sentiment
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Title for the Streamlit App
st.title("Stock Market Dashboard")


# Function to get stock data
@st.cache_data
def get_stock_data(stock, start_date, end_date):
    try:
        data = yf.download(stock, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {stock}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


# ARIMA model for prediction
def arima_prediction(data):
    try:
        # Create progress indicators
        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.text("Fitting ARIMA model...")
        model = ARIMA(data['Close'], order=(5, 1, 0))
        arima_model = model.fit()
        progress_bar.progress(50)

        progress_text.text("Generating forecast...")
        arima_forecast = arima_model.forecast(steps=30)

        # Create future dates for forecast
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data['Close'], label="Actual Closing Price")
        ax.plot(forecast_dates, arima_forecast, label="ARIMA Forecast")
        plt.title("ARIMA Model Stock Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig)

        # Display forecast values
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': arima_forecast
        }).set_index('Date')
        st.write("ARIMA Forecast Values:")
        st.dataframe(forecast_df)

        progress_bar.progress(100)
        progress_text.text("ARIMA prediction completed!")

    except Exception as e:
        st.error(f"Error in ARIMA prediction: {str(e)}")


# LSTM model for prediction
def lstm_prediction(data):
    try:
        # Create progress indicators
        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.text("Preparing data for LSTM...")
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # Prepare training data
        train_size = int(np.ceil(len(scaled_data) * 0.8))
        train_data = scaled_data[0:train_size, :]
        progress_bar.progress(20)

        # Create sequences
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        progress_bar.progress(40)

        progress_text.text("Training LSTM model...")
        # Build and train LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=0)
        progress_bar.progress(70)

        progress_text.text("Generating predictions...")
        # Prepare test data
        test_data = scaled_data[train_size - 60:, :]
        x_test, y_test = [], scaled_data[train_size:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Generate predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        progress_bar.progress(90)

        # Prepare data for plotting
        train = data['Close'][:train_size]
        valid = data['Close'][train_size:]
        valid = pd.DataFrame(valid)
        valid['Predictions'] = predictions

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(train.index, train, label="Training Data")
        ax.plot(valid.index, valid['Close'], label="Actual Price")
        ax.plot(valid.index, valid['Predictions'], label="LSTM Predictions")
        plt.title("LSTM Model Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig)

        # Display predictions
        st.write("LSTM Predictions vs Actual Values:")
        st.dataframe(valid)

        progress_bar.progress(100)
        progress_text.text("LSTM prediction completed!")

    except Exception as e:
        st.error(f"Error in LSTM prediction: {str(e)}")

# Fetch and display news articles with AI summaries and sentiment
def display_news_summary(company_name):
    st.header(f"News Summary and Sentiment for {company_name}")

    # Fetch recent news
    st.subheader("Fetching latest news...")
    articles = get_news(company_name)

    if not articles:
        st.write("No recent news articles found.")
        return

    # Generate AI summaries
    st.subheader("Summarizing news articles...")
    summaries = summarize_news(articles)

    # Perform sentiment analysis
    sentiments = analyze_sentiment(articles)

    # Display summaries and sentiments
    st.subheader("Recent News Summaries and Sentiment Analysis")
    for summary, sentiment in zip(summaries, sentiments):
        st.write(f"**Title**: [{summary['title']}]({summary['link']})")
        st.write(f"**Summary**: {summary['summary']}")
        st.write(f"**Sentiment**: {sentiment['sentiment']} (Confidence: {sentiment['score']:.2f})")
        st.write("---")


def main():
    st.sidebar.header("Input Parameters")

    # Stock selection
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL")

    # Date range selection
    start_date = st.sidebar.date_input(
        "Start Date",
        datetime.date.today() - datetime.timedelta(days=365 * 2)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        datetime.date.today()
    )

    if st.sidebar.button("Analyze Stock"):
        # Fetch data
        with st.spinner('Fetching stock data...'):
            data = get_stock_data(stock_symbol, start_date, end_date)

        if data is not None:
            # Display basic stock info
            st.subheader("Stock Price Overview")
            st.line_chart(data['Close'])

            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["ARIMA Prediction", "LSTM Prediction", "News Summary"])

            with tab1:
                st.header("ARIMA Model Prediction")
                arima_prediction(data)

            with tab2:
                st.header("LSTM Model Prediction")
                lstm_prediction(data)

            with tab3:
                st.header("News Summary and Sentiment Analysis")
                display_news_summary(stock_symbol)  # Call the news summary function

if __name__ == "__main__":
    main()