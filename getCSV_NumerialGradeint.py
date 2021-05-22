import csv
import pandas as pd
import numpy as np

def get_CSV(load):
    data1 = []
    data1 = pd.read_csv(load)
    data2 = data1.to_numpy()
    return data2

def numerial_gradient(f,x):
    h = 1e-4
    grad = np.zeros[x.size]

    for i in range (x.size):
        grad[i] = (f(x[i]+h) - f(x[i]-h)) / 2*h

    return grad

# X_train = get_CSV('X_train.csv')
# X_test = get_CSV('X_test.csv')
# y_train = get_CSV('y_train.cxv')
# y_test = get_CSV('y_test.csv')
# validation = get_CSV('validation.csv')
