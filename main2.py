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
import matplotlib.pyplot as plt


def get_CSV(load):
    data1 = pd.read_csv(load)
    data2 = data1.to_numpy()
    return data2

def plotting_ready(x, y, label, linestyle, color):
    plt.plot(x, y, label=label, linestyle=linestyle, color=color)

def plotting(x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

X_train = get_CSV('X_train.csv')
X_test = get_CSV('X_test.csv')
y_train = get_CSV('y_train.csv')
y_test = get_CSV('y_test.csv')
validation_data = get_CSV('validation.csv')

card_data = pd.read_csv('creditcard.csv')
card_data.Class.value_counts(normalize=True)
print(card_data.Class.value_counts(normalize=True)*100)

x_val = validation_data[:,2:31]
y_val = validation_data[:,31]

x_train = X_train[:,2:]  #index랑 time 빼기
x_test = X_test[:,2:]

x_train_normal = preprocessing.normalize(x_train,norm='l1')
x_test_normal = preprocessing.normalize(x_test,norm='l1')
x_val_normal = preprocessing.normalize(x_val,norm='l1')

# samplng_strategy는 소수 class가 1 : {sampling_strategy}만큼의 비율로 데이터가 커진다는 것.
X, Y = SMOTE(random_state=0,sampling_strategy=0.3).fit_resample(x_train_normal,y_train[:,1])
print("====data_size====")
print("BEFORE = x_train : ",x_train_normal.shape,"y_train: ",y_train[:,1].shape)
print("AFTER = X : ", X.shape, " Y : ", Y.shape)

print("===class_ratio===")
print("BEFORE = 0-num: ",sum(y_train[:,1]==0),"1-num : ",sum(y_train[:,1]==1))
print("AFTER = 0-num : ",sum(Y==0),"1-num : ",sum(Y==1))
#x_t, y_t = SMOTE(sampling_strategy=0.1).fit_resample(x_test_normal,y_test[:,1])

#smote = SMOTE(random_state=0)
#X_train_over,y_train_over = smote.fit_sample(X_train,y_train)
#print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
#print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
#print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())

''' layer를 설정 '''
model1 = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='relu', name='hidden_layer'),
            Dense(1, activation='relu', name='output_layer')]
            )
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
a = model1.fit(x_train_normal,y_train[:,1],epochs=100,batch_size=10000)



plotting_ready(a.epoch, a.history['accuracy'], 'relu,accuracy', '-','r')
plotting_ready(a.epoch, a.history['loss'], 'relu,loss', '--', 'r')



model2 = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='relu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')]
            )
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
b = model2.fit(x_train_normal,y_train[:,1],epochs=100,batch_size=10000)

plotting_ready(b.epoch, b.history['accuracy'], 'sigmoid,accuracy','-','b')
plotting_ready(b.epoch, b.history['loss'], 'sigmoid,loss','--', 'b')



plotting('final layer','acc&loss')

# plt.xlabel('final layer')
# plt.ylabel('acc&loss')
# plt.legend()
# plt.show()
''' layer정보, output shape, param개수 에 대한 정보를 출력 '''
# model1.summary()
# model2.summary()


# ''' back propagation '''
# #model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #categorical_crossentropy [0.0, 0.9985224008560181]
# model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #binary_crossentropy [0.5114834904670715, 0.9985224008560181]
#
# model1.fit(x_train_normal,y_train[:,1],epochs=30,batch_size=10000) #[0.4077814817428589, 0.9985224008560181]
# #model.fit(X,Y,epochs=100,batch_size=X.shape[0]) #[0.4371872544288635, 0.9985224008560181]


# print(model1.evaluate(x_val_normal,y_val,batch_size=1))