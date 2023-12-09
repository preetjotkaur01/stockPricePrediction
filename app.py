#python -m venv C:\Users\preet\Downloads\python\machine\venv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pandas_datareader as data
from keras.models import load_model
from pandas_datareader import data as pdr

import streamlit as st 
# streamlit is used to make the front end of the web application 
# which interacts with the backend python code 
import yfinance as yfin
yfin.pdr_override()
start='2010-01-01'
end='2023-11-20'
st.title('Stock Trend Prediction ')
user_input = st.text_input('Enter Stock Ticker','AAPL ')
df = pdr.get_data_yahoo(user_input, start, end)


#describing data
st.subheader('Data from 2010 - 2023')
st.write(df.describe()) 

#visualisations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100 ,'r')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100 ,'r')
plt.plot(ma200 ,'g')
plt.plot(df.Close)
st.pyplot(fig)

# splitting data into training and testing - 70% data = training and 30% data = testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0,1))
 
data_training_array = scaler.fit_transform(data_training)

#load my model 
model = load_model('keras_model.h5')

#training part 
past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)
X_test = []
Y_test = []
for i in range(100,input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    Y_test.append(input_data[i,0])
    
X_test, Y_test=np.array(X_test),np.array(Y_test)
Y_predicted = model.predict(X_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
Y_predicted = Y_predicted*scale_factor
Y_test = Y_test*scale_factor


#final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(Y_test,'b',label='Original Price')
plt.plot(Y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)