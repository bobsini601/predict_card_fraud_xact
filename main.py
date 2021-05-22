import csv
import pandas as pd
import numpy as np

class PredictFraud :

    def __init__(self):
        X_train = self.get_CSV('X_train.csv')
        X_test = self.get_CSV('X_test.csv')
        y_train = self.get_CSV('y_train.csv')
        y_test = self.get_CSV('y_test.csv')
        validation = self.get_CSV('validation.csv')
        theta1 = np.random.random((15, 31))
        theta2=np.random.random((1,16))
        
    #load the csv file
    def get_CSV(load):
        data1 = []
        data1 = pd.read_csv(load)
        data2 = data1.to_numpy()
        return data2


    def once_numerical_gradient(self, f, x):
        h = 1e-4
        grad = np.zeros[x.size]

        for i in range(x.size):
            grad[i] = (f(x[i] + h) - f(x[i] - h)) / 2 * h

        return grad

    # def numerical_gradient(self, x, t):
    #     loss_W = lambda W: self.loss(x, t)
    # 
    #     grads = {}  # 각 매개변수의 기울기
    #     grads['W1'] = self.numerical_gradient(loss_W, self.params['W1'])
    #     grads['b1'] = self.numerical_gradient(loss_W, self.params['b1'])
    #     grads['W2'] = self.numerical_gradient(loss_W, self.params['W2'])
    #     grads['b2'] = self.numerical_gradient(loss_W, self.params['b2'])
    # 
    #     return grads


    def partial_deriv(self,m, X, Y):
        sum = 0
        for i in range(0, m):
            sum += (self.h(X[i]) - Y[i]) * X[i]
        return sum / m

    def act_func(self,x, alpha=0.5):  # ELU 함수(activate 함수). alpha는 0.5가 적당하다고 함
        return (x > 0) * x + (x <= 0) * (alpha * (np.exp(x) - 1))

    # m은 input data 개수
    # X는 input data, Y는 실제 output
    # (X1, Y1), ..., (Xm, Ym)은 주어지는 것!
    # h()는 output 계산하는 함수

    def h(self,theta, X):  # hypothesis 함수 -> 세타를 전치(t)하고, x를 곱한 다음 활성화함수에 넣기
        t_theta = np.transpose(theta)  # 전치
        return self.act_func(np.dot(t_theta, X),0.5)

    def cost_fn(self,m, X, Y):  # cost 함수 (regularization는 나중에 넣을 것)
        sum = 0
        for i in range(1, m + 1):
            sum += Y[i] * np.log(self.h(X[i])) + (1 - Y[i]) * np.log(self.h(1 - X[i]))
        return (-1 / m) * sum

    '''
     lr : learning rate    step_size : 적절한 크기의 반복 횟수 "loop를 몇 번 할 건지"

     learning rate 같은 경우, 초반에 설정하는 learning rate 값에는 정답이 없다. 
     시작을 0.01로 시작해서 overshooting이 일어나면 Learning rate의 값을 줄이고 
     학습 속도가 매우 느리다면 Learning rate 값을 올리는 방향으로 학습을 진행하면 될 것이다.
    '''

    def gradient_descent(self, lr=0.01, step_size=100):
        # local minimum 으로 빠지지 않게 주의
        # theta가 여러 개인 경우, 그 갯수만큼 loop 안에 넣어야함.

        # step_size 만큼 loop를 돈다.
        for _ in range(step_size):
            # 기울기가 설정해놓은 0에 다다른 경우,멈추기
            if self.partial_deriv(m,X,Y) == 0:
                break
            else:
                self.theta1-=lr*self.partial_deriv(m,X,Y)
                self.theta2-=lr*self.partial_deriv(m,X,Y)

