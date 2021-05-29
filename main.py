import csv
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.pardir)
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from tensorflow.python.keras.models import Sequential, InputLayer
from tensorflow.python.keras.layers import Dense, Activation
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_CSV(load):
    data1 = []
    data1 = pd.read_csv(load)
    data2 = data1.to_numpy()
    return data2

def plot_data_size(values,name):
    x=np.arange(2)
    plt.bar(x,values)
    plt.xticks(np.arange(0,2))
    plt.title(name)
    plt.show()

def plot_scatter_data(X,Y,name):
    scaled_data=StandardScaler().fit_transform(X)
    pc = PCA(n_components=2).fit_transform(scaled_data)
    df = pd.DataFrame(np.c_[pc, Y], columns=['x1', 'x2', 'y'])
    plt.scatter(df['x1'],df['x2'],c=df['y'],cmap=plt.cm.plasma)
    plt.title(name)
    plt.show()

def plotting_ready(x, y, label, linestyle, color):
    plt.plot(x, y, label=label, linestyle=linestyle, color=color)

def plotting(x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


''' load the data files '''
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


''' 
SMOTE the minor data (fraud transaction) 
samplng_strategy는 소수 class가  '1 : {sampling_strategy}' 만큼의 비율로 데이터가 커진다는 것. 
'''
X, Y = SMOTE(sampling_strategy=1).fit_resample(x_train_normal,y_train[:,1])


''' SMOTE 하기 전과 후의 train data와 test data의 크기 '''
print("====data_size====")
print("BEFORE = x_train : ",x_train_normal.shape[0],"\t y_train: ",y_train[:,1].shape[0])
print("AFTER = X : ", X.shape[0], "\t Y : ", Y.shape[0])

''' SMOTE 하기 전과 후의 정상거래 data와 사기거래 data의 크기 '''
print("===class_ratio===")
print("BEFORE = 0-num: ",sum(y_train[:,1]==0),"\t 1-num : ",sum(y_train[:,1]==1))
print("AFTER = 0-num : ",sum(Y==0),"\t 1-num : ",sum(Y==1))


''' before SMOTE, the number of data (class 0, class 1) '''
before_values=[sum(y_train[:,1]==0),sum(y_train[:,1]==1)]
plot_data_size(before_values,'Before SMOTE, class_size')

''' after SMOTE, the number of data (class 0, class 1) '''
after_values=[sum(Y==0),sum(Y==1)]
plot_data_size(after_values,'After SMOTE, class_size')


''' 사기거래 data와 정상거래 data 2차원으로 축소 후 plot '''
plot_scatter_data(x_train_normal,y_train[:,1],'BeforeSMOTE')

''' 사기거래 data와 정상거래 data 2차원으로 축소 후 plot'''
plot_scatter_data(X,Y,'AfterSMOTE')



####################################################################################
# 6.3 COMOPARE RERU, ELU
####################################################################################
relu_model = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='relu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')]
            )
relu_model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
relu_res = relu_model.fit(X,Y,epochs=100,batch_size=1000,validation_data=(x_val_normal,y_val))

plotting_ready(relu_res.epoch, relu_res.history['accuracy'], 'relu,accuracy', '-','r')
plotting_ready(relu_res.epoch, relu_res.history['loss'], 'relu,loss', '--', 'r')

elu_model = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='elu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')]
            )
elu_model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
elu_res = elu_model.fit(X,Y,epochs=100,batch_size=1000,validation_data=(x_val_normal,y_val))

plotting_ready(elu_res.epoch, elu_res.history['accuracy'], 'elu,accuracy', '-','b')
plotting_ready(elu_res.epoch, elu_res.history['loss'], 'elu,loss', '--', 'b')

plotting('final layer','acc&loss')

####################################################################################
# 6.4 FIND OPTIMAL EPOCHS, BATCH_SIZE
####################################################################################
b_list=[100,1000,2000,X.shape[0]]

for b_size in b_list:
    model = Sequential([
        InputLayer(input_shape=(29,)),
        Dense(15, activation='elu', name='hidden_layer'),
        Dense(1, activation='sigmoid', name='output_layer')]
    )

    model.compile(loss='binary_crossentropy', optimizer='RMSprop')  # binary_crossentropy
    batch_res = model.fit(X,Y,epochs=100,batch_size=b_size,validation_data=(x_val_normal,y_val))

    plotting_ready(batch_res.epoch, batch_res.history['accuracy'],b_size, '{0},accuracy'.format(b_size), '-', 'b')
    plotting_ready(batch_res.epoch, batch_res.history['loss'], b_size,'{0},loss'.format(b_size), '--', 'b')

    model.evaluate(x_test_normal, y_test[:, 1], batch_size=x_test_normal.shape[0])


plotting('final layer', 'acc&loss')
