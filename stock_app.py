import dash
from dash import dcc, html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv(r"C:\Users\LENOVO\Desktop\Stock-Price-Prediction-Project-Code\stock_data.csv")

# Convert the Date column to datetime and sort the dataset
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

# Feature and target selection
X = df[["Open", "High", "Low", "Volume"]]  # Feature columns
y = df["Close"]  # Target column

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load the pre-trained model (only load once)
model = load_model(r"C:\Users\LENOVO\Desktop\Stock-Price-Prediction-Project-Code\saved_lstm_model.h5")

# Dash app setup
app = dash.Dash(__name__)

# Define layout of the app
app.layout = html.Div([
    html.H1("Stock Price Prediction Dashboard"),
    dcc.Dropdown(
        id="stock-dropdown",
        options=[{"label": stock, "value": stock} for stock in df["Stock"].unique()],
        value=df["Stock"].unique()[0]  # Default value
    ),
    dcc.Graph(id="price-graph"),
    dcc.Interval(id="interval-component", interval=1 * 60000, n_intervals=0)  # Refresh every minute
])

# Callback to update the graph
@app.callback(
    dash.dependencies.Output("price-graph", "figure"),
    [dash.dependencies.Input("stock-dropdown", "value"),
     dash.dependencies.Input("interval-component", "n_intervals")]
)
def update_graph(selected_stock, n):
    filtered_df = df[df["Stock"] == selected_stock]

    # Predict the future stock prices using the model (example with last 10 rows)
    recent_data = filtered_df[["Open", "High", "Low", "Volume"]].tail(10)
    recent_data_scaled = scaler.transform(recent_data)
    predictions = model.predict(recent_data_scaled)

    # Create the graph
    traces = [
        go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["Close"],
            mode="lines",
            name="Actual Price"
        ),
        go.Scatter(
            x=filtered_df["Date"].tail(10),
            y=predictions.flatten(),
            mode="lines",
            name="Predicted Price"
        )
    ]

    return {
        "data": traces,
        "layout": go.Layout(
            title=f"Stock Prices for {selected_stock}",
            xaxis={"title": "Date"},
            yaxis={"title": "Price"}
        )
    }

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
