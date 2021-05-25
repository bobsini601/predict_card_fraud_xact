import csv
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation


class PredictFraud:

    def __init__(self):
        self.X_train = self.get_CSV('X_train.csv')
        self.X_test = self.get_CSV('X_test.csv')
        self.y_train = self.get_CSV('y_train.csv')
        self.y_test = self.get_CSV('y_test.csv')
        self.validation = self.get_CSV('validation.csv')
        self.create_model()

    # load the csv file
    def get_CSV(load):
        data1 = []
        data1 = pd.read_csv(load)
        data2 = data1.to_numpy()
        return data2


    def create_model(self):
        model = Sequential([
            Dense(15, input_dim=30, input_shape=(159491,30), activation='relu',name='hidden_layer_1'),
            Dense(1,activation='sigmoid',name='output_layer')]
            )
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
        model.fit(self.X_train,self.y_train,epochs=100,batch_size=64,validation_data=(self.X_train,self.y_train))
        print(model.evaluate(self.X_train,self.y_train,batch_size=64))


if __name__=='__main__':
    res1=PredictFraud()