

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import dot


def logit(x):
    return 1./(1+np.exp(-x))



data = pd.read_csv("C:/Users/10365/OneDrive - stevens.edu/CS559 Machine Learning/Homework_1/iris.data", names=['sep len', 'sep wid', 'pet len', 'pet wid', 'class'])


data.replace("Iris-setosa", "Iris-non-virginica", inplace=True)
data.replace("Iris-versicolor", "Iris-non-virginica", inplace=True)


X = data.loc[:, ['sep len', 'sep wid']]
print("X = ", X)
print(type(X))
x = np.array(X)
print("X = ", x)
print(type(x))
print(x.shape)
y = data['class'].values
y = [k.replace("Iris-non-virginica", '0') for k in y]
y = [k.replace("Iris-virginica", '1') for k in y]
y = list(map(int, y))
print("y = ", y)
print(type(y))
y = np.matrix(y)
print("y = ", y)
print(y.shape)

m, n = x.shape  # 矩阵大小
alpha = 0.0065  # 设定学习速率
theta_g = np.zeros((n, 1))  # 初始化参数
maxCycles = 3000  # 迭代次数
J = pd.Series(np.arange(maxCycles, dtype=float))  # 损失函数

for i in range(maxCycles):
    h = logit(dot(X, theta_g))  # 估计值
    J[i] = -(1 / 100.) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))  # 计算损失函数值
    error = h - Y  # 误差
    grad = dot(X.T, error)  # 梯度
    theta_g -= alpha * grad
