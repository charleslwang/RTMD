import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Title for the Streamlit App
st.title("Market Dashboard using ARIMA and LSTM Models")


# Function to get stock data
@st.cache_data
def get_stock_data(stock, start_date, end_date):
    try:
        data = yf.download(stock, start=start_date, end=end_date)
        if 'Adj Close' in data.columns and 'Close' not in data.columns:
            data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()


# Function to prepare data for LSTM
def prepare_lstm_data(data, lookback=60):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create sequences for LSTM
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler


# LSTM model for prediction
def lstm_prediction(data):
    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow is required for LSTM predictions. Please install it using: pip install tensorflow")
        return

    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare data
        status_text.text('Preparing data for LSTM...')
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(data)
        progress_bar.progress(20)

        # Build LSTM model
        status_text.text('Building LSTM model...')
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        progress_bar.progress(40)

        # Train model
        status_text.text('Training LSTM model...')
        model.fit(X_train, y_train,
                  batch_size=32,
                  epochs=20,
                  verbose=0)
        progress_bar.progress(70)

        # Generate predictions
        status_text.text('Generating predictions...')
        last_sequence = data['Close'].values[-60:]
        last_sequence = scaler.transform(last_sequence.reshape(-1, 1))
        next_sequence = last_sequence.reshape(1, 60, 1)

        # Predict next 30 days
        lstm_predictions = []
        current_sequence = next_sequence

        for _ in range(30):
            next_pred = model.predict(current_sequence, verbose=0)
            lstm_predictions.append(next_pred[0, 0])
            current_sequence = np.append(current_sequence[:, 1:, :],
                                         [[next_pred[0, 0]]],
                                         axis=1)

        # Transform predictions back to original scale
        lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
        forecast_dates = pd.date_range(data.index[-1], periods=31)[1:]

        progress_bar.progress(90)
        status_text.text('Creating visualizations...')

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot actual data
        ax.plot(data.index[-100:], data['Close'][-100:], label='Historical Price', color='blue')

        # Plot predictions
        ax.plot(forecast_dates, lstm_predictions, color='green', label='LSTM Forecast')

        plt.title("LSTM Model Stock Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig)

        # Display forecast values
        st.write("\nLSTM Forecast Values:")
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'LSTM Forecast': lstm_predictions.flatten()
        }).set_index('Date')

        # Calculate forecast metrics
        last_price = data['Close'][-1]
        forecast_return = (lstm_predictions[-1] - last_price) / last_price * 100

        st.write("\nLSTM Forecast Metrics:")
        forecast_metrics = {
            "Current Price": f"${last_price:.2f}",
            "Forecast End Price": f"${lstm_predictions[-1][0]:.2f}",
            "Forecasted Return": f"{forecast_return[0]:.2f}%",
            "Forecast Period": "30 days"
        }
        st.write(forecast_metrics)

        # Show detailed forecast table
        st.write("\nDetailed Forecast:")
        st.dataframe(forecast_df.style.format("{:.2f}"))

        progress_bar.progress(100)
        status_text.text('LSTM Analysis complete!')

    except Exception as e:
        st.error(f"Error in LSTM prediction: {str(e)}")


# ARIMA model for prediction
def arima_prediction(data):
    try:
        # Fit ARIMA model
        model = ARIMA(data['Close'], order=(5, 1, 0))
        arima_model = model.fit()

        # Make predictions
        forecast_steps = 90
        forecast = arima_model.forecast(steps=forecast_steps)
        forecast_dates = pd.date_range(data.index[-1], periods=forecast_steps + 1)[1:]

        # Create confidence intervals
        forecast_conf = arima_model.get_forecast(forecast_steps)
        conf_int = forecast_conf.conf_int()

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot actual data
        ax.plot(data.index, data['Close'], label='Historical Price')

        # Plot forecast
        ax.plot(forecast_dates, forecast, color='red', label='ARIMA Forecast')

        # Plot confidence intervals
        ax.fill_between(forecast_dates,
                        conf_int.iloc[:, 0],
                        conf_int.iloc[:, 1],
                        color='red',
                        alpha=0.1,
                        label='95% Confidence Interval')

        plt.title("ARIMA Model Stock Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig)

        # Display forecast values
        st.write("\nForecast Values:")
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast,
            'Lower CI': conf_int.iloc[:, 0],
            'Upper CI': conf_int.iloc[:, 1]
        }).set_index('Date')
        st.write(forecast_df)

    except Exception as e:
        st.error(f"Error in ARIMA prediction: {str(e)}")

# Display basic stock data insights
def stock_data_insights(data):
    st.write("Basic Statistics of Stock Data:")
    st.write(data.describe())

    # Create two columns for charts
    col1, col2 = st.columns(2)

    with col1:
        st.write("Stock Price History")
        st.line_chart(data['Close'])

    with col2:
        # Calculate and plot daily returns
        returns = data['Close'].pct_change()
        st.write("Daily Returns Distribution")
        fig, ax = plt.subplots()
        returns.hist(bins=50, ax=ax)
        st.pyplot(fig)

    # Add more insights
    returns = data['Close'].pct_change()
    st.write("\nStock Performance Metrics:")
    metrics = {
        "Daily Returns Mean": f"{returns.mean():.4f}",
        "Daily Returns Std": f"{returns.std():.4f}",
        "Annual Volatility": f"{returns.std() * np.sqrt(252):.4f}",
        "Total Return": f"{(data['Close'][-1] / data['Close'][0] - 1) * 100:.2f}%",
        "Sharpe Ratio": f"{(returns.mean() / returns.std()) * np.sqrt(252):.2f}"
    }
    st.write(metrics)


# Main function
def main():
    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())

    if st.sidebar.button("Fetch and Predict"):
        # Show loading message
        with st.spinner('Fetching and analyzing data...'):
            data = get_stock_data(stock_symbol, start_date, end_date)

            if data.empty:
                st.error("No data found for the given stock and date range.")
            else:
                # Create tabs for different sections
                tab1, tab2, tab3 = st.tabs(["📈 Stock Analysis",
                                            "🔮 ARIMA Prediction",
                                            "🤖 LSTM Prediction"])

                with tab1:
                    st.header("Stock Analysis")
                    stock_data_insights(data)

                with tab2:
                    st.header("ARIMA Model Prediction")
                    arima_prediction(data)

                with tab3:
                    st.header("LSTM Model Prediction")
                    lstm_prediction(data)


# Run the app
if __name__ == "__main__":
    main()