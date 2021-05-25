import sys, os
import pandas as pd
sys.path.append(os.pardir)
from imblearn.over_sampling import SMOTE
from keras.layers import Dense, Activation
from sklearn import preprocessing
import keras

def get_CSV(load):
    data1 = pd.read_csv(load)
    data2 = data1.to_numpy()
    return data2

X_train = get_CSV('X_train.csv')
y_train = get_CSV('y_train.csv')
X_test = get_CSV('X_test.csv')
y_test = get_CSV('y_test.csv')
validation_data = get_CSV('validation.csv')

x_val = validation_data[:,2:31]
y_val = validation_data[:,31]

x_train = X_train[:,2:]  #index랑 time 빼기
x_test = X_test[:,2:]

x_train_normal = preprocessing.normalize(x_train,norm='l1')
x_test_normal = preprocessing.normalize(x_test,norm='l1')
x_val_normal = preprocessing.normalize(x_val,norm='l1')

sm = SMOTE(sampling_strategy=1)
X, Y = sm.fit_resample(x_train_normal,y_train[:,1])
x_t, y_t = sm.fit_resample(x_test_normal,y_test[:,1])

input = keras.Input(shape=(29,))
x=Dense(15,activation='relu',name='hidden_layer')(input)
output=Dense(1,activation='sigmoid',name='output_layer')(x)

model = keras.Model(input, output)
model.summary()

model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr=1e-3),metrics=['accuracy']) #binary_crossentropy
model.fit(x_train_normal,y_train[:,1],epochs=10,batch_size=x_train_normal.shape[0], validation_data=(x_val_normal,y_val))
# model.summary()
score = model.evaluate(x_test_normal,y_test[:,1],batch_size=1)
print(score)  #loss & accuracy 출력
