#1.1 数据加载--查看鸢尾花数据集
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# 加载鸢尾花数据集
data = load_iris()

# 转换为 DataFrame 方便查看
df = pd.DataFrame(data.data, columns=data.feature_names)    
# cloumns 列  特征名称   data.feature_names：特征的名称列表（比如鸢尾花数据集中的 ['sepal length (cm)', 'sepal width (cm)'] 等）

df['target'] = data.target
# data.target：数据集的标签数据（即输出变量，比如鸢尾花的类别编号 0、1、2），通常是一个一维数组（长度 = 样本数）。

df['species'] = df['target'].apply(lambda x: data.target_names[x])
#data.target_names：标签编号对应不同种类名称（比如鸢尾花中 0→'setosa'、1→'versicolor' 等）

# 查看前几行数据
print(df.head(10))



#1.2 可视化--使用 matplotlib 和 seaborn 库来进行可视化。
import seaborn as sns
import matplotlib.pyplot as plt
# 绘制特征之间的关系
sns.pairplot(df, hue="species")
plt.show()



#1.3 热力图可视化特征之间的相关性
# 绘制特征之间的关系
correlation_matrix = df.drop(columns=['target', 'species']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#2.1 数据预处理
# 标准化 将每个特征转换为零均值和单位方差  目的是使每个特征的均值为 0，方差为 1，这对于一些基于距离的模型（如 KNN、SVM）非常重要。
from sklearn.preprocessing import StandardScaler
X = df.drop(columns=['target','species'])
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#2.2 特征选择
#有时我们需要通过特征选择来减少特征维度，提升模型效果。例如，使用 SelectKBest 选择与标签最相关的 2 个特征：
from sklearn.feature_selection import SelectKBest, f_classif

# 使用卡方检验选择 2 个最相关的特征
selector = SelectKBest(f_classif, k=2)
X_new = selector.fit_transform(X_scaled, y)

# 打印选择的特征
selected_features = selector.get_support(indices=True)
print("Selected features:", X.columns[selected_features])



#3.1 使用决策树分类器
#首先尝试使用决策树（Decision Tree）模型进行分类。
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化决策树分类器
model_dt = DecisionTreeClassifier(random_state=42)

# 训练模型
model_dt.fit(X_train, y_train)

# 预测
y_pred_dt = model_dt.predict(X_test)

# 评估模型
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")


#3.2 使用支持向量机（SVM）进行分类
from sklearn.svm import SVC

# 初始化 SVM 分类器
model_svm = SVC(kernel='linear', random_state=42)

# 训练模型
model_svm.fit(X_train, y_train)

# 预测
y_pred_svm = model_svm.predict(X_test)

# 评估模型
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")



#4.1 模型评估
#除了准确率外，我们还可以使用其他评估指标，如混淆矩阵、精度、召回率和 F1 分数等。
from sklearn.metrics import classification_report, confusion_matrix

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix (Decision Tree):")
print(cm)

# 精度、召回率、F1 分数
report = classification_report(y_test, y_pred_dt)
print("Classification Report (Decision Tree):")
print(report)


#4.2网格搜索调优
#为了优化模型，我们可以使用网格搜索（GridSearchCV）对模型的超参数进行调优，找到最佳的参数组合。
#通过网格搜索，我们可以找到最适合当前数据的决策树参数，并提升模型的预测准确率。
from sklearn.model_selection import GridSearchCV

# 定义决策树的参数网格
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 初始化 GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5)

# 训练网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数和最佳模型
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 预测和评估
y_pred_optimized = best_model.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"Optimized Decision Tree Accuracy: {accuracy_optimized:.4f}")


#4.3交叉验证
from sklearn.model_selection import cross_val_score

# 进行 5 折交叉验证
cross_val_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print(f"Cross-validation Scores: {cross_val_scores}")
print(f"Mean CV Accuracy: {cross_val_scores.mean():.4f}")