# Market Sales Analysis and Forecasting Dasboard with LSTM Model

## Data Description
The dataset used for this project is the "Kalimati Tarkari Fruits and Vegetables Price" dataset from Kaggle, which can be found [here](https://www.kaggle.com/datasets/usmanlovescode/kalimati-tarkari-fruits-and-vegetables-price). The dataset includes the following columns:
- `Date`: The date of the record
- `Commodity`: The type of fruit or vegetable
- `Minimum Price`: The minimum price recorded on that day
- `Maximum Price`: The maximum price recorded on that day
- `Average Price`: The average price recorded on that day

## Data Cleaning
Data cleaning steps include:
- Handling missing values and outliers
- Aggregating data to daily average prices
- Getting commodities series data to analyis
- Formatting the date column to a datetime object

## Exploratory Data Analysis
The EDA includes:
- Visualizing price trends and seasonality over time for various commodities 
- Analyzing the distribution of prices
- Identifying patterns and anomalies in the data

## Modeling
The modeling phase includes:
- Building an LSTM model for time series forecasting
- Evaluating model performance using metrics such as Mean Squared Error (MSE) 
- By comparing the estimated and actual values using a line plot that displays the predicted and actual prices over time, the performance of the model is assessed.


To run the project, follow these steps:
1. Clone the repository:
   ```sh
   git clone https://github.com/jonathanlex1/market_sales_time_series.git
2. Navigate to the project directory 
    ```sh
    cd market_sales_time_series
3. Run the streamlit app
    ```sh
    streamlit run app.py 
