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
X, Y = SMOTE(sampling_strategy=0.1).fit_resample(x_train_normal,y_train[:,1])
X2, Y2= SMOTE(sampling_strategy=0.5).fit_resample(x_train_normal,y_train[:,1])
X3, Y3= SMOTE(sampling_strategy=1).fit_resample(x_train_normal,y_train[:,1])
X4, Y4= SMOTE(sampling_strategy=0.3).fit_resample(x_train_normal,y_train[:,1])

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



''' Set the layers '''
model = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='elu',name='hidden_layer_1'),
            Dense(1,activation='sigmoid',name='output_layer')
            ])

''' layer정보, output shape, parameter 개수에 대한 정보를 출력 '''
model.summary()

''' back propagation ( loss fn, optimizer(=gradient descent) ) '''
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

''' 학습시키는 과정 epochs = 반복횟수, batch_size = 학습할 데이터의 크기 '''
hist=model.fit(x_train_normal,y_train[:,1],epochs=100,batch_size=1000)    #SMOTE 적용 안했을 때
#model.fit(X,Y,epochs=100,batch_size=2000)
plotting_ready(hist.epoch, hist.history['accuracy'], 'NOSMOTE,accuracy', '-','black')
plotting_ready(hist.epoch, hist.history['loss'], 'NOSMOTE,loss', '--', 'black')

''' evaluate the loss, accuracy '''
print(model.evaluate(x_test_normal,y_test[:,1],batch_size=x_test_normal.shape[0]))


####################################################################################
model1 = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='elu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')]
            )
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
a = model1.fit(X,Y,epochs=100,batch_size=1000)


plotting_ready(a.epoch, a.history['accuracy'], '0.1,accuracy', '-','g')
plotting_ready(a.epoch, a.history['loss'], '0.1,loss', '--', 'g')


model4 = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='elu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')]
            )
model4.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
c = model4.fit(X4,Y4,epochs=100,batch_size=1000)


plotting_ready(c.epoch, c.history['accuracy'], '0.3,accuracy', '-','y')
plotting_ready(c.epoch, c.history['loss'], '0.3,loss', '--', 'y')


model2 = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='elu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')]
            )
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
b = model2.fit(X2,Y2,epochs=100,batch_size=1000)

plotting_ready(b.epoch, b.history['accuracy'], '0.5,accuracy','-','b')
plotting_ready(b.epoch, b.history['loss'], '0.5,loss','--', 'b')


model3 = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='elu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')]
            )
model3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
c = model3.fit(X3,Y3,epochs=100,batch_size=1000)


plotting_ready(c.epoch, c.history['accuracy'], '1,accuracy', '-','r')
plotting_ready(c.epoch, c.history['loss'], '1,loss', '--', 'r')

#############################################################################################
print("1",model3.evaluate(x_test_normal,y_test[:,1],batch_size=x_test_normal.shape[0]))
print("0.5",model2.evaluate(x_test_normal,y_test[:,1],batch_size=x_test_normal.shape[0]))
print("0.3",model4.evaluate(x_test_normal,y_test[:,1],batch_size=x_test_normal.shape[0]))
print("0.1",model1.evaluate(x_test_normal,y_test[:,1],batch_size=x_test_normal.shape[0]))

plotting('final layer','acc&loss')
