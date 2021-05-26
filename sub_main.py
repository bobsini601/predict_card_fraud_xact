import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from tensorflow.python.keras.models import Sequential, InputLayer
from tensorflow.python.keras.layers import Dense, Activation
import keras
from matplotlib import pyplot as plt


def get_CSV(load):
    data1 = []
    data1 = pd.read_csv(load)
    data2 = data1.to_numpy()
    return data2

''' load the data files'''
X_train = get_CSV('X_train.csv')
X_test = get_CSV('X_test.csv')
y_train = get_CSV('y_train.csv')
y_test = get_CSV('y_test.csv')
validation_data = get_CSV('validation.csv')

x_val = validation_data[:,2:31]
y_val = validation_data[:,31]

''' delete index, time '''
x_train = X_train[:,2:]
x_test = X_test[:,2:]

''' normalize the x_train, x_test, x_val '''
x_train_normal = preprocessing.normalize(x_train,norm='l1')
x_test_normal = preprocessing.normalize(x_test,norm='l1')
x_val_normal = preprocessing.normalize(x_val,norm='l1')

''' SMOTE the minor data (fraud transaction) '''
# samplng_strategy는 소수 class가 1 : {sampling_strategy}만큼의 비율로 데이터가 커진다는 것.
X, Y = SMOTE(random_state=4,sampling_strategy=0.3).fit_resample(x_train_normal,y_train[:,1])
print("====data_size====")
print("BEFORE = x_train : ",x_train_normal.shape,"y_train: ",y_train[:,1].shape)
print("AFTER = X : ", X.shape, " Y : ", Y.shape)

print("===class_ratio===")
print("BEFORE = 0-num: ",sum(y_train[:,1]==0),"1-num : ",sum(y_train[:,1]==1))
print("AFTER = 0-num : ",sum(Y==0),"1-num : ",sum(Y==1))

''' before SMOTE, the number of data (class 0, class 1) '''
y=np.arange(2)
values=[sum(y_train[:,1]==0),sum(y_train[:,1]==1)]
plt.bar(y,values)
plt.show()

''' after SMOTE, the number of data (class 0, class 1) '''
x=np.arange(2)
values=[sum(Y==0),sum(Y==1)]
plt.bar(x,values)
plt.show()



#print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())

''' Set the layers '''
model = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='relu',name='hidden_layer_1'),
            Dense(1,activation='sigmoid',name='output_layer')]
            )

''' layer정보, output shape, param개수 에 대한 정보를 출력 '''
model.summary()

''' back propagation ( loss fn, optimizer(=gradient descent) )'''
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #categorical_crossentropy [0.0, 0.9985224008560181]
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #binary_crossentropy [0.5114834904670715, 0.9985224008560181]

''' 학습시키는 과정 epochs = 반복횟수, batch_size = 학습할 데이터의 크기'''
#model.fit(x_train_normal,y_train[:,1],epochs=100,batch_size=x_train_normal.shape[0]) #[0.4077814817428589, 0.9985224008560181]
model.fit(X,Y,epochs=100,batch_size=X.shape[0]) #[0.4371872544288635, 0.9985224008560181]
print("X.shape[0] : ",X.shape[0]) # 206957

''' evaluate the loss, accuracy '''
print(model.evaluate(x_test_normal,y_test[:,1],batch_size=1000))