import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing  #feature normalize


def elu(x,alpha): # ELU 함수(activate 함수). alpha는 0.5가 적당하다고 함
    return (x>0)*x + (x<=0)*(alpha*(np.exp(x)-1))

def h(theta, X): # hypothesis 함수 -> 세타를 전치(t)하고, x를 곱한 다음 활성화함수에 넣기
    return elu(np.dot(theta, X),0.5)

def get_CSV(load):
    data1 = pd.read_csv(load)
    data2 = data1.to_numpy()
    return data2

X_train = get_CSV('X_train.csv')
# X_test = get_CSV('X_test.csv')
# y_train = get_CSV('y_train.csv')
# y_test = get_CSV('y_test.csv')
# validation = get_CSV('validation.csv')

x_train = X_train[:,2:]
x_train_normal = preprocessing.normalize(x_train,norm='l1')

# normalize 이후 plot
# for i in range(0,29):
#     plt.subplot(5,6,i+1)  #동시표기
#     plt.hist(x_train_normal[:,i],bins=10)
#     plt.title("V"+str(i+1))
#     if (i==28):
#         plt.hist(x_train_normal[:,i],bins=10)  #bins=10 <=> 막대개수=10
#         plt.title("amount")
# plt.subplots_adjust(hspace=1)  #간격 조정
# plt.show()

print(X_train[0].shape)
print(X_train.shape)

theta1 = np.random.random((15,29))
print(theta1.shape)

theta2 = np.random.random((1,16))
print(theta2.shape)

a2 = []
for i in x_train_normal:
    a2.append(h(theta1,i))

x = a2
ones = np.array([[1]] * len(x))
new_a2 = np.append(ones, x, axis=1)
print(new_a2)

a3=[]
for i in new_a2:
    a3.append(h(theta2, i))

print(a3[0].shape)
print(a3)