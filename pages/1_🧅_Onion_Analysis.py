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

st.title('🧅 Onion Sales Analysis')  