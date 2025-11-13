# 线性回归 (Linear Regression) 是一种用于预测连续值的最基本的机器学习算法，
# 它假设目标变量 y 和特征变量 x 之间存在线性关系，并试图找到一条最佳拟合直线来描述这种关系。
# y = w * x + b    y 是预测值、x 是特征变量、w 是权重 (斜率)、b 是偏置 (截距)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


np.random.seed(0)
x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.rand(100,1)


# plt.scatter(x,y)
# plt.xlabel("x")
# plt.ylabel('y')
# plt.title('Generated Date From Runoob')
# plt.show()


#创建线性回归模型
model = LinearRegression()

#拟合模型(训练)
model.fit(x,y)

#评估
score = model.score(x,y)
print("得分：",score)

#输出模型的参数
print(f"斜率 (w):{model.coef_[0][0]}")
print(f"截距 (b):{model.intercept_[0]}")

#预测
y_pred = model.predict(x)

plt.scatter(x,y)
plt.xlabel("x")
plt.ylabel('y')
plt.plot(x, y_pred, color = 'red')
plt.title('LinearRegression Fit')
plt.show()

