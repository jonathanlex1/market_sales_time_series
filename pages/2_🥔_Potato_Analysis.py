import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


from keras.models import load_model

import altair as alt

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid') 

st.set_page_config(page_title="Potato Analysis ", page_icon="📈")

#import data 
df = pd.read_csv('./data/cleaned_market_sales.csv')

#main page 
st.title('🥔 Potato Sales Analysis')

st.subheader('Prices of All Potatoes')

#line chart 
potato_red = df[df['commodity'] == 'Potato Red']
potato_indian = df[df['commodity'] == 'Potato Red(Indian)']
potato_mude = df[df['commodity'] == 'Potato Red(Mude)']

combined = pd.concat([potato_red, potato_indian, potato_mude])

chart = alt.Chart(combined).mark_line().encode(
    x='date:T',
    y='avg:Q',
    color = 'commodity:N'
)

st.altair_chart(chart, use_container_width=True)
st.markdown("""
##### Insight : 
- During the period when data for all types is available, prices for "Potato Red (Indian)" and "Potato Red (Mude)" tend to move in line with "Potato Red", but slightly lower.
- In 2020, there was a significant increase in prices for each type of potato and a decline in early 2021.
"""
)

#heatmap 
potatoes_corr = df[df['commodity'].str.contains('Potato')].pivot_table(index='date', columns='commodity', values='avg').corr()
fig,ax = plt.subplots()

sns.heatmap(potatoes_corr, cmap='coolwarm', annot=True, ax=ax)
st.pyplot(fig)

st.markdown("""
#### Insight : 
- The prices of Potato Red, Indian Red Potato, Russet Potato, and Potato White show strong correlation.
"""
)
