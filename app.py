import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

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

#sitebar
# with st.sidebar : 
#     st.subheader('Sales')

#     option = st.selectbox(
#     "How would you like to be contacted?",
#     ("Onion", "Potato", "Ginger"),
#     )

#     if st.button('app')
#     if option == 'Onion' : 
#         st.switch_page('./pages/onion_pg.py')
#     elif option == 'Potato' :
#         st.switch_page('./pages/potato_pg.py')
#     elif option == 'Ginger' : 
#         st.switch_page('./pages/ganger_pg.py')


#import data 
df = pd.read_csv('./cleaned_market_sales.csv')
data_series = pd.read_csv("./daily_sales_data.csv")

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
st.subheader('High and Low Demand Commodities')

fig, ax = plt.subplots(1,2, figsize=(12,5))

df['commodity'].value_counts().head(10).plot(kind='barh', title = 'Top 10 Most Quantities Of Commodity', ax=ax[0],color= 'dodgerblue')
df['commodity'].value_counts().tail(10).plot(kind='barh', title = 'Top 10 Less Quantities Of Commodity', ax=ax[1],color= 'dodgerblue')
ax[1].invert_yaxis()
plt.tight_layout()

st.pyplot(fig)
st.write('Insight :')
st.markdown('- The plots show Onion Dry (Indian) as the highest quantity commodity, followed by Brd Leaf Mustard, Banana, various vegetables, and spices like Chilli Dry and Ginger. The lowest quantities include Onion Dry (Chinese), fruits like Mandarin and Mango (Dushari), and items like "Maize" and Sweet Lime. For high-quantity items, efficient inventory management and promotions are key to avoiding overstock. For low-quantity items, strengthen supply chains and form partnerships with local producers. Marketing should highlight the benefits of high-quantity items and create exclusivity for low-quantity ones. Using data analytics for demand forecasting and diversifying the product range with premium varieties can help balance supply and attract more customers.')

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
    st.table(data)

st.write('Insight : ')
st.markdown('- The distribution plot shows significant price variation, with most prices below 250, indicating that most commodities are affordably priced and driven by supply and demand for lower-priced items. However, the long tail suggests some commodities have higher prices, indicating a smaller market segment for these higher-priced items.')

st.markdown("- Commodities such as Ginger, Cabbage (Local), Raddish White (Local), Potato Red, Bamboo Shoot, Brd Leaf Mustard, Onion Dry (Indian), and Banana have high demand due to their low prices.")

# Assuming df is your DataFrame
st.subheader('Commodities Prices over Year')

fig, ax = plt.subplots(figsize=(12, 5))

df[df['commodity'] == 'Onion Dry (Indian)'].plot(x='date', y='avg', ax=ax, label='Onion Dry (Indian)')
df[df['commodity'] == 'Cabbage(Local)'].plot(x='date', y='avg', ax=ax, label='Cabbage(Local)')
df[df['commodity'] == 'Asparagus'].plot(x='date', y='avg', ax=ax, label='Asparagus')
df[df['commodity'] == 'Banana'].plot(x='date', y='avg', ax=ax, label='Banana')
df[df['commodity'] == 'Brd Leaf Mustard'].plot(x='date', y='avg', ax=ax, label='Brd Leaf Mustard')
df[df['commodity'] == 'Potato Red'].plot(x='date', y='avg', ax=ax, label='Potato Red')

ax.set_title('Prices for Most Commodities Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Price')

st.pyplot(fig)
st.write('Insight : ')
st.markdown('- The price of asparagus is significantly higher compared to other commodities.')
st.markdown('- Asparagus was only available from 2013 to 2016, whereas other commodities continued to be available until 2020 at lower prices.')

#daily sales
st.subheader('Daily Sales 2013-2015')
fig,ax = plt.subplots(figsize=(12,5))
data_series.plot(x='date', y='avg',ax=ax,color= 'dodgerblue')
st.pyplot(fig)

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

st.line_chart(df_predictions)
st.table(df_predictions)




