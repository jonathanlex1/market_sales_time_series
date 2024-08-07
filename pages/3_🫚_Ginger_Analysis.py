import streamlit as st
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import altair as alt
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Ginger", page_icon="📈")

st.title('🫚 Ginger Sales Analysis')   

ginger_series = pd.read_csv('./data/ginger_series.csv')

chart = alt.Chart(ginger_series).mark_line().encode(
    x='date:T',
    y='avg:Q'
)

st.altair_chart(chart, use_container_width=True)


st.subheader('Forecasting Ginger Prices')

number = st.select_slider(
    "Select a daily number from 1 - 30",
    options=[
        1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30
    ],
)

scaler = MinMaxScaler(feature_range=(-1,1))
model = load_model('./model/ginger_lstm_model.h5')


if not pd.api.types.is_datetime64_any_dtype(ginger_series.index):
    ginger_series.index = pd.to_datetime(ginger_series.index)
data = ginger_series[-30:]
data = data.set_index('date')
data_scaled = scaler.fit_transform(data[['avg']])
data_input = data_scaled.reshape((1,len(data),1))

predictions = []

for i in range(number) : 
  prediction = model.predict(data_input)
  prediction_actual = scaler.inverse_transform(prediction)
  predictions.append(prediction_actual[0,0])
  data_input = np.append(data_input[:, 1:, :], prediction.reshape(1,1,1), axis=1)

start_date = data.index[-1]

if isinstance(start_date, str):
    start_date = pd.to_datetime(start_date)

date = [start_date + timedelta(days=x) for x in range(1, len(predictions)+1)]

prediction_df = pd.DataFrame(
    {'date' : date,
    'predictions' : predictions}
)
prediction_df = prediction_df.set_index('date')

tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
tab1.line_chart(prediction_df)
tab2.write(prediction_df)