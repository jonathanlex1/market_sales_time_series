import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

import altair as alt

from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')    

st.set_page_config(
    page_title="Market Sales Analysis",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded")

st.title('🏪 Market Sales Dashboard')

#import data 
df = pd.read_csv('./data/cleaned_market_sales.csv')
data_series = pd.read_csv("./data/daily_sales_data.csv")

total_sales = sum(df['avg'])
total_min_sales = sum(df['min'])
total_max_sales = sum(df['max'])

col1,col2,col3 = st.columns(3)
with col1 : 
    st.metric(label = 'Total Avarage Sales', value=f'{total_sales:,.2f}')
with col2 : 
    st.metric(label = 'Total Min Sales', value=f'{total_min_sales:,.2f}')
with col3 :
    st.metric(label = 'Total Max Sales', value=f'{total_max_sales:,.2f}')

##High and Low Demand Commodities

commodity_counts = df['commodity'].value_counts()
ten_most = commodity_counts.head(10).sort_values(ascending=False)
ten_less = commodity_counts.tail(10).sort_values(ascending=False)

col1 , col2 = st.columns(2)
with col1 :
    st.subheader('Top 10 Commodities by Total') 
    st.bar_chart(ten_most,horizontal=True)
with col2 :
    st.subheader('Bottom 10 Commodities by Total')
    st.bar_chart(ten_less,horizontal=True)

st.markdown("""
##### Insight : 
- The plots show Onion Dry (Indian) as the highest quantity commodity, with Brd Leaf Mustard, Banana, and various vegetables and spices also in high supply. The lowest quantities include Onion Dry (Chinese), Mandarin, Mango (Dushari), Maize, and Sweet Lime. Efficient inventory management and promotions are key for high-quantity items, while strengthening supply chains and partnerships is essential for low-quantity items. Marketing should emphasize the benefits of high-quantity items and create exclusivity for low-quantity ones. Using data analytics for demand forecasting and diversifying the product range can help balance supply and attract customers.""")

##Prices Distrubtion 
st.subheader('Prices Distribution')
col1,col2 = st.columns(2)

with col1 :
    
    fig,ax = plt.subplots(figsize=(5,5))
    hisplot= sns.histplot(data=df['avg'], kde=True, ax=ax,color= 'dodgerblue')

    st.pyplot(fig)  

with col2 : 
    ## low prices commodities 
    st.write('Commodities With Low Prices')
    data = df[df['avg'] < 250].value_counts('commodity').head(10)  
    st.write(data)

st.markdown("""
##### Insight : 
- The distribution plot shows significant price variation, with most prices below 250, indicating that most commodities are affordably priced and driven by supply and demand for lower-priced items. However, the long tail suggests some commodities have higher prices, indicating a smaller market segment for these higher-priced items. 
- Commodities such as Ginger, Cabbage (Local), Raddish White (Local), Potato Red, Bamboo Shoot, Brd Leaf Mustard, Onion Dry (Indian), and Banana have high demand due to their low prices.
            """)

# Assuming df is your DataFrame
st.subheader('Commodities Prices over Year')


onion_dry_indian = df[df['commodity'] == 'Onion Dry (Indian)']
cabbage_local = df[df['commodity'] == 'Cabbage(Local)']
asparagus = df[df['commodity'] == 'Asparagus']
banana = df[df['commodity'] == 'Banana']
brd_leaf = df[df['commodity'] == 'Brd Leaf Mustard']
potato_red = df[df['commodity'] == 'Potato Red']

combined = pd.concat([onion_dry_indian,cabbage_local,asparagus,banana,brd_leaf,potato_red])

chart = alt.Chart(combined).mark_line().encode(
    x = 'date:T',
    y= 'avg:Q',
    color='commodity:N'
)
st.altair_chart(chart, use_container_width=True)

st.markdown("""
##### Insight : 
- The price of asparagus is significantly higher compared to other commodities.
- Asparagus was only available from 2013 to 2016, whereas other commodities continued to be available until 2020 at lower prices.""")

#daily sales
st.subheader('Daily Sales 2013-2021')

chart = alt.Chart(data_series).mark_line().encode(
    x= 'date:T',
    y= 'avg:Q',
)
st.altair_chart(chart, use_container_width=True)

## Forecasting
st.subheader('Forecasting Daily Sales')
number = st.select_slider(
    "Select a daily number from 1 - 30",
    options=[
        1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30
    ],
)

#minxmax scaler
scaler = MinMaxScaler(feature_range=(-1,1))

if not pd.api.types.is_datetime64_any_dtype(data_series.index):
    data_series.index = pd.to_datetime(data_series.index)

# #load model 
lstm_model = load_model('./model/lstm_model.h5')

#getting 30 last data for predict 
data_series = data_series.set_index('date')
last_datas = data_series[-30:]
last_datas_scaled = scaler.fit_transform(last_datas[['avg']])
input_data = last_datas_scaled.reshape((1,30,1))

#prediction new values in the future
predictions = []

for i in range(number) : 
    predicted_value = lstm_model.predict(input_data)
    predicted_actual = scaler.inverse_transform(predicted_value)
    predictions.append(predicted_actual[0,0])

    predicted_actual_reshaped = predicted_actual.reshape(1,1,1)
    input_data = np.append(input_data[:, 1:, :],predicted_actual_reshaped, axis=1)

#making dataframe and concatenate new date and prediction
start_date = last_datas.index[-1]

if isinstance(start_date, str):
    start_date = pd.to_datetime(start_date)

date = [start_date + timedelta(days=x) for x in range(1, len(predictions)+1)]

df_predictions = pd.DataFrame(
    {'date' : date,
     'prediction' : predictions}
)

df_predictions = df_predictions.set_index('date')

tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
tab1.line_chart(df_predictions)
tab2.write(df_predictions)




