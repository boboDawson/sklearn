#逻辑回归（Logistic Regression）是一种广泛应用于分类问题的统计学习方法，尽管名字中带有"回归"，但它实际上是一种用于二分类或多分类问题的算法。
#逻辑回归通过使用逻辑函数（也称为 Sigmoid 函数）将线性回归的输出映射到 0 和 1 之间，从而预测某个事件发生的概率。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  #模型选择与评估的模块
from sklearn.linear_model import LogisticRegression  #线性模型模块
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  #模型评估指标模块 accuracy_score：计算准确率、confusion_matrix：生成混淆矩阵、classification_report：生成分类报告

#加载数据
iris = load_iris()
#只使用前两个特征   切片
X = iris.data[: , :2]  # 花萼长度、花萼宽度
#将目标转化成二分类问题  布尔判断（筛选类别）、* 1：布尔值转数值（二分类标签）
y = (iris.target !=0) * 1
# X特征、y标签
#划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

#创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
# y_test：测试集的「真实标签」、y_pred：模型的「预测标签」（模型通过 X_test 计算出的预测结果）
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率:{accuracy:.2f}") # 字符串前缀 f 表示「格式化字符串」

# 混淆矩阵  一个 n×n 的二维数组（n 是分类任务的类别数），称为「混淆矩阵」，行代表「真实类别」，列代表「预测类别」。
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵")
print(conf_matrix)

# 分类报告
class_report = classification_report(y_test, y_pred)
print("分类报告")
print(class_report)


# 可视化决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1    # 花萼长度,最大值最小值+-1是留边距
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1    # 花萼宽度
# np.meshgrid()：将一维坐标转为二维网格
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),    # 生成 x 轴的连续坐标
                     np.arange(y_min, y_max, 0.01))    # 生成 y 轴的连续坐标

# 将密集网格点的预测结果转化为可绘图格式  np.c_[a, b]：按「列」拼接两个一维数组，生成二维特征矩阵（核心！）
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])       # ravel() 方法：把多维数组「扁平化」为一维数组（不改变数据顺序，仅改变形状）
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap='BuGn_r', alpha = 0.8) # cmap=颜色映射(也可以不要)，alpha=透明度
# plt.contour(xx ,yy, Z, levels=[0.5] ) # levels=0.5 是决策边界（概率=0.5）
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o")
plt.xlabel('Sepal lemgth')
plt.ylabel('Sepal widyh')
plt.title('Logistic Regression Decision Boundary')
plt.show()


# 补充
"""只想展示决策边界（如逻辑回归的分类线）	plt.contour()
想展示概率分布 / 数值梯度（如正类概率从低到高）	plt.contourf()
图像背景复杂，需避免遮挡原始数据	plt.contour()
报告 / 演示，需直观展示区域差异	plt.contourf()
"""