# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
num_observations = 500

#创建X的正负样本
x1 = np.random.multivariate_normal([0,0],[[1,.75],[.75,1]], num_observations)
x2 = np.random.multivariate_normal([1,4],[[1,.75],[.75,1]], num_observations)

X = np.vstack((x1, x2)).astype(np.float32)

# 创建相应的正负标签
y = np.hstack((-np.ones(num_observations), np.ones(num_observations)))
plt.subplot(121)

# 在图中标出正负样本
for d, sample in enumerate(X):     

    if d < 500:         
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2, color='purple')         
    else:        
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2, color='yellow')


# 优化函数
def perceptron_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1 # 学习率
    epochs = 20 # 迭代轮数
    errors = []
    
    for t in range(epochs):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0: #仅当分类错误时，即f(x)和y异号时才更新w
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
        errors.append(total_error*-1)
    
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    
    return w
plt.subplot(122)
w = perceptron_sgd(X,y)

# 作出分类超平面
plt.subplot(121)
x2 = [w[0], w[1], -w[1], w[0]]
x3 = [w[0], w[1], w[1], -w[0]]
x2x3 = np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')
print(w)
