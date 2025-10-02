import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint

# ----------------------------
# Streamlit App
# ----------------------------
st.title("📈 Stock Price Prediction")
st.write("Predict stock prices using historical data + 5-day forecast.")

# User inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOG, MSFT)", "GOOG")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

if st.button("Run Prediction"):

    # ----------------------------
    # 1. Load & preprocess data
    # ----------------------------
    st.write("### Loading Data...")
    df = yf.download(ticker, start=start_date, end=end_date)
    st.write(df.tail())

    features = ["Open", "High", "Low", "Close", "Volume"]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features].values)

    # Train / Validation / Test split
    train_size = int(len(scaled_data) * 0.7)
    val_size = int(len(scaled_data) * 0.2)
    test_size = len(scaled_data) - train_size - val_size

    data_train = scaled_data[:train_size]
    data_val = scaled_data[train_size:train_size + val_size]
    data_test = scaled_data[train_size + val_size:]

    # ----------------------------
    # 2. Helper: sequence builder
    # ----------------------------
    def construct_lstm_data(data, seq_len, target_idx):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i, :])
            y.append(data[i, target_idx])
        return np.array(X), np.array(y)

    sequence_length = 60
    target_idx = features.index("Close")

    X_train, y_train = construct_lstm_data(data_train, sequence_length, target_idx)
    all_data = np.concatenate([data_train, data_val, data_test], axis=0)

    X_val, y_val = construct_lstm_data(
        all_data[train_size - sequence_length:train_size + val_size], sequence_length, target_idx
    )

    X_test, y_test = construct_lstm_data(
        all_data[-(test_size + sequence_length):], sequence_length, target_idx
    )

    # ----------------------------
    # 3. Build / Train Model
    # ----------------------------
    #st.write("### Training Model...")
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    best_model_path = "best_lstm_model.h5"
    checkpoint = ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, mode="min")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[checkpoint],
        verbose=0
    )

    model = load_model(best_model_path)

    # ----------------------------
    # 4. Predictions
    # ----------------------------
    st.write("### Model Evaluation")
    preds = model.predict(X_test)

    # Inverse transform
    y_test_full = []
    preds_full = []
    for i in range(len(preds)):
        dummy_true = np.zeros(len(features))
        dummy_true[target_idx] = y_test[i]
        y_test_full.append(scaler.inverse_transform([dummy_true])[0][target_idx])

        dummy_pred = np.zeros(len(features))
        dummy_pred[target_idx] = preds[i]
        preds_full.append(scaler.inverse_transform([dummy_pred])[0][target_idx])

    result_df = pd.DataFrame({"Actual": y_test_full, "Predicted": preds_full}, index=df.index[-len(y_test):])
    st.line_chart(result_df)

    # ----------------------------
    # 5. Forecast next 5 days
    # ----------------------------
    st.write("### Next 5-Day Forecast")

    future_input = scaled_data[-sequence_length:]
    future_input = future_input.reshape(1, sequence_length, len(features))
    future_preds = []

    for _ in range(5):
        pred = model.predict(future_input, verbose=0)[0]

        dummy = np.zeros(len(features))
        dummy[target_idx] = pred
        inv_pred = scaler.inverse_transform([dummy])[0][target_idx]
        future_preds.append(inv_pred)

        new_row = future_input[0, -1, :].copy()
        new_row[target_idx] = pred
        future_input = np.append(future_input[:, 1:, :], [[new_row]], axis=1)

    last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq="B")

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_preds})
    st.write(forecast_df)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result_df.index, result_df["Actual"], label="Actual")
    ax.plot(result_df.index, result_df["Predicted"], label="Predicted")
    ax.plot(forecast_df["Date"], forecast_df["Predicted_Close"], "ro--", label="Forecast (5 Days)")
    ax.legend()
    st.pyplot(fig)
