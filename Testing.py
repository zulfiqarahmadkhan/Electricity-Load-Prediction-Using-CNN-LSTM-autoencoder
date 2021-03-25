# -*- coding: utf-8 -*-
"""
This work has been published in MDPI sensors journal
the title of the paper is "Towards Efficient Electricity Forecasting in Residential and Commercial Buildings: A Novel Hybrid CNN with a LSTM-AE based Framework"
"""
#importamt Liberaries 
import pandas as pd
from numpy import array
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

#MAPE calculation
def mean_absolute_percentage_error(predicted, y_test):
    y_test, predicted = np.array(y_test), np.array(predicted)
    return np.mean(np.abs((y_test-predicted)/y_test)) *100
#Data and lable generator funtion
def split(data, max_values):
    Y = []
    X=[]
    cnt=max_values/8
    cnt=int(cnt)
    c=0
    for i in range (cnt):
        
        X.append(data[c:c+8])
        Y.append(data[c+8:c+12])
        c+=1

    return X,Y

#Total number of record in the dataset
max_values=138352

#Reading the dataset
df = pd.read_csv('dataset/IHEPC.csv', sep=',', 
                 parse_dates={'dt' : ['datetime']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'], index_col='dt', 
                 usecols = ['datetime', 'Global_active_power'])

# filling nan with mean in any columns
droping_list_all=[]
for j in range(0,1):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)        
        #print(df.iloc[:,j].unique())
for j in range(0,1):        
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
        
# another sanity check to make sure that there are not more any nan
df.isnull().sum()
trainvalues = df.values

#Normalized the values in to a specific range
scaler = MinMaxScaler(feature_range=(0, 1))
data = np.array(trainvalues).reshape(-1,1)
data = scaler.fit_transform(data)
trainvalues = np.array(data).reshape(-1)

#call Data and lable generator funtion 
X, Y=split(trainvalues, max_values)
X=array(X)

#Splitting the data into training and testing 
X, testX, Y, testY = train_test_split(X, Y, test_size=0.33, random_state=42)

#Reshaping the data for the proposed model 
trainX = np.array(X).reshape((X.shape[0], X.shape[1], 1))
trainY = np.array(Y)
trainY = np.array(trainY).reshape(trainY.shape[0], trainY.shape[1] )
testX = np.array(testX).reshape((testX.shape[0], testX.shape[1], 1))
testY = np.array(testY)
testY = np.array(testY).reshape(testY.shape[0], testY.shape[1] )

#Loading the model
model = load_model('model/model.h5')

#Testing the model 
pred = model.predict(testX, verbose=0)
testY=testY.flatten()
test_output=pred.flatten()

#ploting the firt 100 acutal and predicted values   
plt.plot(testY[:100], label='Actual power', marker='.') 
plt.plot(test_output[:100], label='Predicted power', marker='.')
plt.savefig('results/actual_predicted.png')
plt.show()

#Model evaluation over MSE, MAE, RMSE, and MAPE
mse=mean_squared_error(testY, test_output)
rmse=sqrt(mean_squared_error(testY, test_output))
mae=mean_absolute_error(testY, test_output)
MAPE=mean_absolute_percentage_error(testY, test_output)
print( 'mse', mse, 'mae', mae, 'rmse', rmse, 'mape', MAPE)
