import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split  # 模型选择
from sklearn.metrics import accuracy_score            # 模型评估

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[: , :2]    # 只用前两个特征
y = iris.target    # 获取数据集的 类别标签（目标变量）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

# 创建SVM分类器
model_SVM = svm.SVC(kernel='linear')   # 使用线性核函数

# 训练模型
model_SVM.fit(X_train, y_train)

# 再测试集上预测
y_pred = model_SVM.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率：{accuracy:.2f}")

# 绘制决策边界
def plot_decision_boundary(X, y, model):
    h = .02  # 网格步长
    x_min, x_max= X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max= X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(X_train, y_train, model_SVM)