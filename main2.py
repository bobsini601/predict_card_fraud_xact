import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def elu(x,alpha): # ELU 함수(activate 함수). alpha는 0.5가 적당하다고 함
    return (x>0)*x + (x<=0)*(alpha*(np.exp(x)-1))

def h(theta, X): # hypothesis 함수 -> 세타를 전치(t)하고, x를 곱한 다음 활성화함수에 넣기
    return elu(np.dot(theta, X),0.5)

# for i in range(1,11):
#     print('['+str(i)+','+str(j)+']', end=",")
#     print('[' + str(3) + ',' + str(3) + ']', end=",")

# a = np.array([1,2,3])
# print(a.shape)
# print(np.ndim(a))
# b = np.array([[1],[2],[3]])
# print(b.shape)
# print(np.ndim(b))

# a=np.array([[1],[2],[3],[4],[5],[6]])
# print(np.transpose(a)) #[[1 2 3 4 5 6]]

def get_CSV(load):
    data1 = pd.read_csv(load)
    data2 = data1.to_numpy()
    return data2

X_train = get_CSV('X_train.csv')
x_train = X_train[:,2:]
# X_test = get_CSV('X_test.csv')
# y_train = get_CSV('y_train.csv')
# y_test = get_CSV('y_test.csv')
# validation = get_CSV('validation.csv')

# print(X_train[0])
print(X_train[0].shape)
print(X_train.shape)

theta1 = np.random.random((15,29))
# print(theta1)
print(theta1.shape)

theta2 = np.random.random((1,15))
# print(theta2)
print(theta2.shape)

a2 = []
for i in x_train:
    a2.append(h(theta1,i))

print(a2[0].shape)

a3=[]
for i in a2:
    a3.append(h(theta2, i))

print(a3[0].shape)
print(a3)