import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LogisticRegression:

    def __init__(self, lr=0.05, num_iter=3, fit_intercept=True,
                 verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        self.theta = np.zeros(X.shape[1])
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / len(y)
            self.theta -= self.lr * gradient
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()


df = pd.read_csv("C:/Users/10365/OneDrive - stevens.edu/CS559 Machine Learning/Homework_1/iris.data", names=['sep len', 'sep wid', 'pet len', 'pet wid', 'class'])

# df.columns = ['sep len', 'sep wid', 'pet len', 'pet wid', 'class']
# print(df['class'].unique())
df.replace("Iris-setosa", "Iris-non-virginica", inplace=True)
df.replace("Iris-versicolor", "Iris-non-virginica", inplace=True)
# print(df['class'].unique())
# print(df)

X = df.loc[:, ['sep len', 'sep wid']]
print("X = ", X)
y = df['class'].values
y = [w.replace("Iris-non-virginica", '0') for w in y]
y = [w.replace("Iris-virginica", '1') for w in y]
y = list(map(int, y))
print("y = ", y)
model = LogisticRegression(lr=0.1, num_iter=300000)
model.fit(X, y)
preds = model.predict(X)
(preds == y).mean()
model.theta
plt.figure(figsize=(10, 6))
plt.scatter(X.loc[:, 'sep len'], X.loc[:, 'sep wid'], color='b')
x1_min, x1_max = X.loc[:, 'sep len'].min(), X.loc[:, 'sep len'].max(),
x2_min, x2_max = X.loc[:, 'sep wid'].min(), X.loc[:, 'sep wid'].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');
plt.show()
