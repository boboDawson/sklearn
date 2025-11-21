from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier   # 决策树分类器的核心类
from sklearn.tree import export_graphviz          # 「导出决策树结构」的工具函数
import graphviz                                   # 导入外部可视化库


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

# 创建决策树分类器
model_tree = DecisionTreeClassifier()

# 训练模型
model_tree.fit(X_train, y_train)

# 对测试集进行预测
y_pred = model_tree.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率:{accuracy:.2f}")

# 到处决策树为dot文件
dot_data = export_graphviz(model_tree, out_file=None,
                           feature_names=iris.feature_names,
                           class_names=iris.target_names,
                           filled=True, rounded=True,
                           special_characters=True)

# 使用graphviz渲染决策树
graph = graphviz.Source(dot_data)
# 保存为PDF文件
graph.render("iris_Decision Tree")
# 在浏览器查看
graph.view()