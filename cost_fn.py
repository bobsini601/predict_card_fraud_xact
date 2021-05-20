import numpy as np
import matplotlib.pyplot as plt

def elu(x,alpha):
    return (x>0)*x + (x<=0)*(alpha*(np.exp(x)-1))

# def elu(x,alpha):
#     if x <= 0:
#         return alpha*(np.exp(x)-1)
#     else:
#         return x

# x = np.arange(-5,5,0.001)
# y = elu(x,0.5)
# plt.plot(x,y)
# plt.show()

x = np.arange(1,31,1) #[1,2,...,30]
# print(x.shape)
theta = np.arange(0.1,3.1,0.1) #[0.1,0.1,...,3.0]
# print(theta.shape)
print(theta)



def cost_fn():
    pass