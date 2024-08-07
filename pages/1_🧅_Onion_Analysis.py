import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model

import altair as alt

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid') 

st.set_page_config(page_title="Onion Analysis ", page_icon="📈")


st.title('🧅 Onion Sales Analysis')  

df = pd.read_csv('./data/cleaned_market_sales.csv')
onion_series = pd.read_csv('./data/onion_series.csv')

st.subheader('Onion Dry (Indian) vs Onion Dry (Chinese)')

indian_onion = df[df['commodity'] == 'Onion Dry (Indian)']
chinese_onion = df[df['commodity'] == 'Onion Dry (Chinese)']
onion_green = df[df['commodity'] == 'Onion Green']

combined = pd.concat([indian_onion, chinese_onion,onion_green])

chart = alt.Chart(combined).mark_line().encode(
    x='date:T',
    y='avg:Q',
    color = 'commodity:N'
)

st.altair_chart(chart, use_container_width=True)
st.markdown("""
    ##### Insight :
    - A significant spike in the price of Onion Dry (Indian) in 2016 and 2020 and a spike in the price of Onion Dry (Chinese) 
    - Onion greens tend to be more expensive than others and have frequent price spikes
            
"""
)
st.subheader('Onions Prices Correlation')
onions = df[df['commodity'].str.contains('Onion')]
onions_corr = onions.pivot_table(index='date', columns='commodity', values='avg').corr()
fig, ax = plt.subplots()
sns.heatmap(onions_corr, cmap='coolwarm', annot=True, ax=ax)
st.pyplot(fig)
st.markdown("""
##### Insight : 
- The prices of Onion Dry (Chinese) and Onion Dry (Indian) show strong correlation.

"""
)

#making prediction for onion prices

st.subheader('Forecasting Onion Prices')

scaler = MinMaxScaler(feature_range=(-1,1))

lstm_model_onion = load_model('./model/onion_lstm_model.h5')

if not pd.api.types.is_datetime64_any_dtype(onion_series.index):
    onion_series.index = pd.to_datetime(onion_series.index)

data = onion_series[-30:]
data = data.set_index('date')

data_scaled = scaler.fit_transform(data[['avg']])
input_data = data_scaled.reshape((1,len(data),1))

predictions = [] 

number = st.select_slider(
    "Select a daily number from 1 - 30",
    options=[
        1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30
    ],
)

for i in range(number) : 
    prediction = lstm_model_onion.predict(input_data) 
    prediction_actual = scaler.inverse_transform(prediction)

    predictions.append(prediction_actual[0,0])

    ##update input_date with the new prediction 
    input_data = np.append(input_data[:, 1:, :], prediction.reshape(1,1,1), axis=1)

#making prediction dataframe 
start_date = data.index[-1]

if isinstance(start_date, str):
    start_date = pd.to_datetime(start_date)

date = [start_date + timedelta(days=x) for x in range(1, len(predictions)+1)]

prediction_df = pd.DataFrame({
    'date' : date ,
    'prediction' : predictions
})

prediction_df = prediction_df.set_index('date')

tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
tab1.line_chart(prediction_df)
tab2.write(prediction_df)

