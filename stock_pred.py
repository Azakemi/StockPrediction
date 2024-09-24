import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

df = pd.read_csv(r"C:\Users\LENOVO\Desktop\Stock-Price-Prediction-Project-Code\Quote-Equity-TATASTEEL-EQ-22-09-2023-to-22-09-2024.csv")

df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
df = df.dropna(subset=["Date"])

df.set_index('Date', inplace=True)
data = df.sort_index(ascending=True)
new_dataset = pd.DataFrame(data={"Close": data["close"]})

final_dataset = new_dataset.values
train_data = final_dataset[:987]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
lstm_model = Sequential()
lstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=100, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stopping = EarlyStopping(monitor='loss', patience=5)
lstm_model.fit(x_train_data, y_train_data, epochs=50, batch_size=32, verbose=2, callbacks=[early_stopping])

inputs_data = new_dataset[len(new_dataset) - len(final_dataset) + 60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = lstm_model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

valid_data = new_dataset[987:].copy()
valid_data['Predictions'] = closing_price.flatten()
valid_data['Return'] = valid_data['Predictions'].pct_change()
valid_data['Investment Decision'] = np.where(valid_data['Return'] > 0.01, 'Buy', 'Hold')

plt.figure(figsize=(16, 8))
plt.plot(new_dataset, label="Historical Data", color='blue', linewidth=2)
plt.plot(valid_data["Close"], label="Validation Data (Actual)", color='green', linewidth=2)
plt.plot(valid_data["Predictions"], label="Predicted Close Price", color='orange', linewidth=2)

buy_signals = valid_data[valid_data['Investment Decision'] == 'Buy']
plt.scatter(buy_signals.index, buy_signals['Predictions'], marker='^', color='g', label='Buy Signal', s=100)

plt.axvline(x=valid_data.index[0], color='red', linestyle='--', linewidth=1, label='Start of Prediction')
for signal_date in buy_signals.index:
    plt.axvline(x=signal_date, color='red', linestyle=':', linewidth=0.5)

plt.title("Stock Price Prediction with Investment Decisions")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xlim(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31'))
plt.legend()
plt.grid()
plt.show()
num_future_days = 90
last_data = scaled_data[-60:].reshape(1, -1, 1)

future_predictions = []
for _ in range(num_future_days):
    next_prediction = lstm_model.predict(last_data)
    future_predictions.append(next_prediction[0, 0])
    last_data = np.append(last_data[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
last_date = df.index.max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_future_days)

future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Predicted Price'])

plt.figure(figsize=(16, 8))
plt.plot(new_dataset, label="Historical Data", color='blue', linewidth=2)
plt.plot(valid_data["Close"], label="Validation Data (Actual)", color='green', linewidth=2)
plt.plot(valid_data["Predictions"], label="Predicted Close Price", color='orange', linewidth=2)
plt.plot(future_df, label="Future Predictions", linestyle='dashed', color='purple', linewidth=2)

plt.scatter(buy_signals.index, buy_signals['Predictions'], marker='^', color='g', label='Buy Signal', s=100)

# Add red dotted lines for clear visual cues
plt.axvline(x=valid_data.index[0], color='red', linestyle='--', linewidth=1, label='Start of Prediction')
for signal_date in buy_signals.index:
    plt.axvline(x=signal_date, color='red', linestyle=':', linewidth=0.5)

plt.title("Stock Price Prediction with Future Projections")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xlim(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31'))
plt.legend()
plt.grid()
plt.show()

print("\nFuture Predictions Summary:")
for date, price in zip(future_dates, future_predictions):
    print(f"Date: {date.strftime('%Y-%m-%d')}, Predicted Price: {price[0]:.2f}")

investment_amount = float(input("Enter the amount you want to invest: "))
investment_time = int(input("Enter the number of days you want to invest for: "))

predicted_return = valid_data['Return'].mean() * investment_time

total_return = investment_amount * (1 + predicted_return)

print("\nInvestment Summary:")
print("Investment Amount: {:.2f}".format(investment_amount))
print("Investment Time: {} days".format(investment_time))
print("Predicted Return: {:.2%}".format(predicted_return))
print("Total Return: {:.2f}".format(total_return))

if predicted_return > 0.01 and np.std(valid_data['Return']) < 0.02:
    print("\nInvestment Advice: Strong Buy and hold for {} days with low volatility.".format(investment_time))
elif predicted_return > 0.01:
    print("\nInvestment Advice: Buy and hold for {} days with moderate volatility.".format(investment_time))
else:
    print("\nInvestment Advice: Hold or sell with high volatility.")
