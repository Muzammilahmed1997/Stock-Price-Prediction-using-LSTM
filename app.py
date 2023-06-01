import pandas as pd   
import matplotlib.pyplot as plt
import plotly.express as px
import time
import numpy as np
from keras.models import load_model
import streamlit as st
import warnings


# Importing Dataset using Pandas.
data = pd.read_csv("AAPL_Stock_Data.csv")

st.title("Stock Price Trend Prediction")

user_input = st.text_input("Enter Stock Name", 'AAPL')

st.subheader('Data from 2010 - 2019')
st.write(data.describe())

st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(data.close)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


st.subheader('100 Days Moving Average vs Time')
mov_average_100 = data.close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(mov_average_100)
plt.plot(data.close)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


# Splitting Data into Train - Test Set

training_data = pd.DataFrame(data['close'][0:int(len(data)*0.70)])    # Creating training dataset with 70% of total values starting from 0.
testing_data = pd.DataFrame(data['close'][int(len(data)*0.70): int(len(data))])      #Remaining 30% pass into test data


from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range= (0,1))
training_array = scaling.fit_transform(training_data)




#load my model
model = load_model('STK_model.h5')

previous_100_days = training_data.tail(100)
final_data = previous_100_days.append(testing_data, ignore_index = True)

input_data = scaling.fit_transform(final_data)

testX = []
testy = []

for n in range(150, input_data.shape[0]):
    testX.append(input_data[n - 150 : n])
    testy.append(input_data[n,0])

# Now converting trainX and trainy into numpy array.
testX, testy = np.array(testX) , np.array(testy)

y_pred = model.predict(testX)

scaler = scaling.scale_

factor = 1/scaler[0]
y_pred = y_pred * factor
testy = testy * factor
y_pred = y_pred.flatten()


# Final Graph
st.subheader('Predictions VS Original')
fig2 = plt.figure(figsize= (12,6))
plt.plot(testy, 'g', label = 'Original Price')
plt.plot(y_pred, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


testing_array = scaling.fit_transform(testing_data)
x_input = testing_array[278:].reshape(1,-1)

temp_input = list(x_input)
temp_input = temp_input[0].tolist()

# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)




data2 = pd.read_csv("AAPL_Stock_Data.csv")
data2 = data2[['close']]


data_array = scaling.fit_transform(data2)

day_new= np.arange(1,101)
next_30days_pred= np.arange(101,131)  


st.subheader('Next 30 Days Prediction')
fig4 = plt.figure(figsize= (12,6))
plt.plot(day_new,scaling.inverse_transform(data_array[1158:]))
plt.plot(next_30days_pred,scaling.inverse_transform(lst_output))
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)