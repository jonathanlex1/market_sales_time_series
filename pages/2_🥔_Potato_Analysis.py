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

st.set_page_config(page_title="Potato Analysis ", page_icon="📈")

#import data 
df = pd.read_csv('./cleaned_market_sales.csv')

#main page 
st.title('🥔 Potato Sales Analysis')

