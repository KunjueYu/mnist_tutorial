import numpy as np

# a = np.array([[1,2,3], [2,3,4]])
# print(a.ndim, a.shape, a.size, a.dtype, type(a))

# b = np.zeros((3,4))
# c = np.ones((3,4))
# d = np.random.randn(2,3)
# e = np.array([[1,2], [2,3], [3,4]])
# f = b*2 - c*3
# g = 2*c*f
# h = np.dot(a,e)
# i = d.mean()
# j = d.max(axis=1)
# k = a[-1][:2]

# x = np.arange(2, 10, 0.2)

import matplotlib.pyplot as plt
# plt.plot(x, x**1.5*.5, 'r-', x, np.log(x)*5, 'g--', x, x, 'b.')
# plt.show()
from sklearn.datasets import fetch_mldata

# download and read mnist
mnist = fetch_mldata('MNIST original', data_home='./')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target


# # print the first image of the dataset
# img1 = X[0].reshape(28, 28)
# plt.imshow(img1, cmap='gray')
# plt.show()

# # print the images after simple transformation
# img2 = 1 - img1
# plt.imshow(img2, cmap='gray')
# plt.show()

# img3 = img1.transpose()
# plt.imshow(img3, cmap='gray')
# plt.show()

# split data to train and test (for faster calculation, just use 1/10 data)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000,random_state=0)

# TODO:use logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
clf = LogisticRegression().fit(X_train, Y_train)

Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(Y_train, Y_train_pred)

test_accuracy = accuracy_score(Y_test,Y_test_pred)
print('logistic regression')
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

# TODO:use naive bayes
from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB().fit(X_train, Y_train)

Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(Y_train, Y_train_pred)

test_accuracy = accuracy_score(Y_test,Y_test_pred)


print('naive bayes')
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

# TODO:use support vector machine
from sklearn.svm import LinearSVC

clf = LinearSVC().fit(X_train, Y_train)

Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(Y_train, Y_train_pred)

test_accuracy = accuracy_score(Y_test,Y_test_pred)



print('support vector machine')
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

# TODO:use SVM with another group of parameters
clf = LinearSVC(C=10.0, intercept_scaling=1, max_iter=4000, random_state=0, tol=1e-05).fit(X_train, Y_train)

Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)

train_accuracy = accuracy_score(Y_train, Y_train_pred)

test_accuracy = accuracy_score(Y_test,Y_test_pred)



print('SVM with another group orftf parameters')
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))