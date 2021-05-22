import numpy as np
import matplotlib.pyplot as plt

def act_func(x,alpha): # ELU 함수(activate 함수). alpha는 0.5가 적당하다고 함
    return (x>0)*x + (x<=0)*(alpha*(np.exp(x)-1))

def sigmod(x): # 시그모이드 함수(그냥 만들어봄 얘도 activate 함수)
    return 1/(np.exp(-x)+1)

x = np.arange(-5,5,0.001)
y = elu(x,0.5)
plt.plot(x,y)
plt.show() #elu 함수 그려보기

# x = np.arange(-5,5,0.001)
# y = sigmod(x)
# plt.plot(x,y)
# plt.show() #sigmod 함수 그려보기

# 여기는 그냥 예로 값 넣어보려고 만든 배열들인데 안쓸듯 잘못만든것같아..
x = np.arange(1,31,1) #[1,2,...,30]
print(x.shape)

w = np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],
             [2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],
             [1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])
print(w.shape)

z = np.dot(x,w)
print(z.shape)

a = elu(z, 0.5)
# a = sigmod(z)
print(a.shape)
# print(a)



# m은 input data 개수
# X는 input data, Y는 실제 output
# (X1, Y1), ..., (Xm, Ym)은 주어지는 것!
# h()는 output 계산하는 함수


def h(theta, X): # hypothesis 함수 -> 세타를 전치(t)하고, x를 곱한 다음 활성화함수에 넣기
    t_theta = np.transpose(theta) #전치
    return act_func(np.dot(t_theta, X))

def cost_fn(m, X, Y): # cost 함수 (regularization는 나중에 넣을 것)
    sum = 0
    for i in range(1,m+1):
        sum += Y[i]*np.log(h(X[i])) + (1-Y[i])*np.log(h(1-X[i]))
    return (-1/m)*sum
