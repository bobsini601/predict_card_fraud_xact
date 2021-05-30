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
import keras.metrics

''' csv 파일을 load 해서 numpy 배열로 바꿔주는 함수 '''
def get_CSV(load):
    data1 = pd.read_csv(load)
    data2 = data1.to_numpy()
    return data2

''' 
class 별로 data size를 plot 해주는 함수이다. (막대그래프 형태_2개 class) 
data size는 class 별로 values 배열에 저장되어있다.
'''
def plot_data_size(values,name):
    x=np.arange(2)
    plt.bar(x,values)
    plt.xticks(np.arange(0,2))
    plt.title(name)
    plt.show()

'''
30개의 input feature를 2개의 feature로 scaling한 뒤에, PCA함수를 통해 2차원으로 줄인다.
'''
def plot_scatter_data(X,Y,name):
    scaled_data=StandardScaler().fit_transform(X)
    pc = PCA(n_components=2).fit_transform(scaled_data)
    df = pd.DataFrame(np.c_[pc, Y], columns=['x1', 'x2', 'y'])
    plt.scatter(df['x1'],df['x2'],c=df['y'],cmap=plt.cm.plasma)
    plt.title(name)
    plt.show()

def plotting_ready(x, y, label, linestyle='-', color='r'):
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


'''accuracy, precision, recall 선언'''
METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall')
]

####################################################################################
# 6.3 COMOPARE ReLU, ELU
####################################################################################
relu_model = Sequential([
            InputLayer(input_shape=(29,)),
            Dense(15, activation='relu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')]
            )
relu_model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
relu_res = relu_model.fit(X,Y,epochs=100,batch_size=2000,validation_data=(x_val_normal,y_val))

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
c_list=['r','g','b','black']

for i in range(4):
    model = Sequential([
        InputLayer(input_shape=(29,)),
        Dense(15, activation='relu', name='hidden_layer'),
        Dense(1, activation='sigmoid', name='output_layer')]
    )

    model.compile(loss='binary_crossentropy', optimizer='RMSprop',metrics=['accuracy'])  # binary_crossentropy
    batch_res = model.fit(X,Y,epochs=100,batch_size=b_list[i],validation_data=(x_val_normal,y_val))
    acc_label='batch{0},accuracy'.format(b_list[i])
    loss_label='batch{0},loss'.format(b_list[i])
    plotting_ready(batch_res.epoch, batch_res.history['accuracy'],acc_label, '-', c_list[i])
    plotting_ready(batch_res.epoch, batch_res.history['loss'],loss_label, '--', c_list[i])

    model.evaluate(x_test_normal, y_test[:, 1], batch_size=x_test_normal.shape[0])


plotting('final layer', 'acc&loss')

####################################################################################
# 6.5 Accuracy plotting according to SMOTE ratio
####################################################################################
x_list=[0.3, 1.0]
dic_x = {0:x_train_normal}
dic_y = {0:y_train[:,1]}
for i in range(2):
    n1, n2 = SMOTE(random_state=0,sampling_strategy=x_list[i]).fit_resample(x_train_normal,y_train[:,1])
    dic_x[i+1] = n1
    dic_y[i+1] = n2

x_list.insert(0,0)

color = ['r','g','b']
test_f1= []
for j in range(3):
    model = Sequential([
        InputLayer(input_shape=(29,)),
        Dense(15, activation='relu', name='hidden_layer'),
        Dense(1, activation='sigmoid', name='output_layer')]
    )

    n_inputs = x_train.shape[1]
    n_output = 2

    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=METRICS)  # binary_crossentropy
    aa = model.fit(dic_x[j], dic_y[j], epochs=100, batch_size=2000, validation_data=(x_val_normal, y_val))
    # model.summary()
    score = model.evaluate(x_test_normal, y_test[:, 1],batch_size=x_test_normal.shape[0])
    test_f1.append(2 * (score[2] * score[3]) / (score[2] + score[3]))
    list_Train = []
    list_val = []
    test = []
    for i in range(100):
        list_Train.append(2 * (aa.history['precision'][i] * aa.history['recall'][i]) / (
                    aa.history['precision'][i] + aa.history['recall'][i]))
        list_val.append(2 * (aa.history['val_precision'][i] * aa.history['val_recall'][i]) / (
                aa.history['val_precision'][i] + aa.history['val_recall'][i]))


    plotting_ready(aa.epoch, list_Train, 'train smote='+str(x_list[j]), '-', color[j])
    plotting_ready(aa.epoch, list_val, 'val smote=' + str(x_list[j]), '--', color[j])

plotting('epoch', 'f1_score')

####################################################################################
# 6.6 FIND OPTIMAL LEARNING RATE
####################################################################################
'''learing rate 비교'''
def diff_lr(learing_rate):  # 0.000001 ~ 1.0
    model = Sequential([
        InputLayer(input_shape=(29,)),
        Dense(15, activation='relu', name='hidden_layer'),
        Dense(1, activation='sigmoid', name='output_layer')]
    )
    
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=learing_rate), metrics=['accuracy'])
    training = model.fit(X, Y, epochs=100, batch_size=2000, validation_data=(x_val_normal, y_val))
    model.summary()
    score = model.evaluate(x_test_normal, y_test[:, 1], batch_size=x_test_normal.shape[0])
    print(score)  # loss & accuracy 출력

    return training

num=0.00001
for i in range(1,7):
    a = diff_lr(num)
    plotting_ready(a.epoch, a.history['loss'], num)
    num*=10
    
plotting('epoch', 'loss')
    
####################################################################################
# 6.7 FIND OPTIMIZER FUNCTION
####################################################################################
opt_dict = {"adam":1, "SGD":2, "RMSprop":3}
def diff_optimizer(opt):  #adam, SGD, RMSprop
    model = Sequential([
        InputLayer(input_shape=(29,)),
        Dense(15, activation='relu', name='hidden_layer'),
        Dense(1, activation='sigmoid', name='output_layer')]
    )
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    training = model.fit(X, Y, epochs=100, batch_size=2000, validation_data=(x_val_normal, y_val))
    model.summary()
    score = model.evaluate(x_test_normal, y_test[:, 1], batch_size=x_test_normal.shape[0])
    print(score)  # loss & accuracy 출력

    return training

for i in opt_dict:
    a = diff_optimizer(i)
    plotting_ready(a.epoch, a.history['loss'], i+", loss")

plotting('epoch', 'loss')

####################################################################################
# 7.1 final result F1_score
####################################################################################

model = Sequential([
    InputLayer(input_shape=(29,)),
    Dense(15, activation='relu', name='hidden_layer'),
    Dense(1, activation='sigmoid', name='output_layer')]
)

n_inputs = x_train.shape[1]
n_output = 2

model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=METRICS)  # binary_crossentropy
aa = model.fit(X, Y, epochs=100, batch_size=2000, validation_data=(x_val_normal, y_val))
score = model.evaluate(x_test_normal, y_test[:, 1],batch_size=x_test_normal.shape[0])
list_Train = []
list_val = []
for i in range(100):
    list_Train.append(2 * (aa.history['precision'][i] * aa.history['recall'][i]) / (
                aa.history['precision'][i] + aa.history['recall'][i]))
    list_val.append(2 * (aa.history['val_precision'][i] * aa.history['val_recall'][i]) / (
            aa.history['val_precision'][i] + aa.history['val_recall'][i]))

plt.subplot(221)
plotting_ready(aa.epoch, aa.history['precision'], label='train_precision', color='r')
plotting_ready(aa.epoch, aa.history['val_precision'], label='val_precision', linestyle='--', color='r')
plt.xlabel('epoch')
plt.ylabel('precision')
plt.legend()

plt.subplot(222)
plotting_ready(aa.epoch, aa.history['recall'], label='train_recall', color='b')
plotting_ready(aa.epoch, aa.history['val_recall'], label='val_recall', linestyle='--', color='b')
plt.xlabel('epoch')
plt.ylabel('recall')
plt.legend()


plt.subplot(223)
plotting_ready(aa.epoch, list_Train, label='train f1 score', color='g')
plotting_ready(aa.epoch, list_val, label='val f1 score', linestyle='--', color='g')
plt.xlabel('epoch')
plt.ylabel('f1 score')
plt.legend()

plt.show()
