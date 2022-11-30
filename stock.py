import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as Back
import datetime
import sys
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.preprocessing   import MinMaxScaler
import tensorflow as tf
from keras.utils import get_custom_objects
import yfinance as yfin
from datetime import date

#Activation Calculated In Dimmaxontinuity
class StockManager:

    def __init__(self):
        self.stock = "KO"
        self.activation = self.tanlu
        self.startDate  = "1970"
        self.data       = None
        self.model     = None
        self.outputs    = None
        self.model      = None
        get_custom_objects().update({'tanlu': Activation(self.tanlu)})
    def tanlu(self,x):
        return Back.maximum(Back.tanh(x),x)
    def setStock(self,name):
        self.stock = name

    def setStockTrainingRange(self,start,end):
        self.startDate = start
        self.endDate   = end
    def getStock(self):
        self.data=yfin.download(self.stock,start=self.startDate+"-01-01",end=date.today().strftime("%Y-%m-%d"))
        self.data = self.data[['Adj Close', 'Open', 'High','Low',"Close","Volume"]]

    def setActivation(self,acti):
        self.activation = acti
    def buildModel(self,x):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=100,activation=self.activation,return_sequences=True,dtype='float32')))
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(GRU(units=60,activation=self.activation,return_sequences=True,dtype='float32')))
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(GRU(units=60,activation=self.activation,return_sequences=True,dtype='float32')))
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(GRU(units=60,activation=self.activation,return_sequences=True,dtype='float32')))
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(SimpleRNN(units=20,activation=self.activation,return_sequences=True,dtype='float32')))
        self.model.add(Dropout(0.4))
        self.model.add(Bidirectional(SimpleRNN(units=20,activation=self.activation,return_sequences=True,dtype='float32')))
        self.model.add(Dropout(0.4))
        self.model.add(Bidirectional(SimpleRNN(units=20,activation=self.activation,return_sequences=True,dtype='float32')))
        self.model.add(Dropout(0.4))
        self.model.add(Bidirectional(SimpleRNN(units=20,activation=self.activation,return_sequences=True,dtype='float32')))
        self.model.add(Dropout(0.4))
        self.model.add(Bidirectional(SimpleRNN(units=20,activation=self.activation,return_sequences=False,dtype='float32')))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=2,activation = 'sigmoid'))
    def trainModel(self):
        stock_data_orig = self.data[self.startDate:self.endDate]
        stock_data_test = self.data[self.endDate:]
        stock_data_orig_len = len(stock_data_orig)
        stock_data_test_len = len(stock_data_test)
        
        # mmaxscale the data 
        mmax = MinMaxScaler(feature_range=(0,1))
        stock_data_orig_minmax_mmaxscaled = mmax.fit_transform(stock_data_orig)
        
        # create training data of s samples and t time steps 
        time_steps=1
        for_periods = 2
        x_train = [] 
        y_train = [] 
        for i in range(time_steps, stock_data_orig_len-1):
            x_train.append(stock_data_orig_minmax_mmaxscaled[i-time_steps:i, 0].astype(float))
            y_train.append(stock_data_orig_minmax_mmaxscaled[i:i+for_periods, 0].astype(float))
        x_train, y_train = np.array(x_train), np.array(y_train)
        # Reshaping x_train for efficient modelling 
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
        
        inputs = pd.concat([stock_data_orig,stock_data_test])
        inputs = inputs.values
        
        # Preparing x_test 
        x_test = [] 
        for i in range(time_steps, stock_data_test_len + time_steps - for_periods):
            x_test.append(inputs[i-time_steps:i,0])
        x_test = np.array(x_test)
        prev = x_test
        x_test = mmax.fit_transform(x_test)
        enc = LabelEncoder()
        print(self.data.columns.to_list())
        self.buildModel(x_train)
        self.model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
        self.model.fit(x_train,y_train,batch_size=16,epochs=300,verbose=1)
        self.model.summary()
        pred = self.model.predict(x_test)
        pred = mmax.inverse_transform(pred)
        self.model.save("./trained.h5")
        plt.xlabel('Date',fontsize=10)
        plt.ylabel('Close',fontsize=10)
        plt.plot(prev[:,0],'r',pred[:,0],'g')
        plt.show()
        
if __name__ == '__main__' :
    st=StockManager()
    st.setStock(sys.argv[1])
    st.setStockTrainingRange("2010","2021")
    st.getStock()
    st.trainModel()
