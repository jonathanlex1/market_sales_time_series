import streamlit as st
import pandas as pd 
import matplotlib as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')    

#import data 

st.set_page_config(
    page_title="Market Sales Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded")

st.title('🏪 Market Sales Dashboard')

df = pd.read_csv('C:\Streamlit Learning\Time Series Sales Market\market_sales_time_series\cleaned_market_sales.csv')

total_sales = sum(df['avg'])
total_min_sales = sum(df['min'])
total_max_sales = sum(df['max'])

col1,col2,col3 = st.columns(3)
with col1 : 
    st.metric(label = 'Total Sales', value=f'{total_sales:,.2f}')
with col2 : 
    st.metric(label = 'Total Min Sales', value=f'{total_min_sales:,.2f}')
with col3 :
    st.metric(label = 'Total Max Sales', value=f'{total_min_sales:,.2f}')
