# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

iris_data = datasets.load_iris()
X = iris_data.data[:, [2, 3]]
y = iris_data.target
label_list = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# 这就是个画图函数...
def plot_decision_regions(X, y, classifier=None):
    marker_list = ['o', 'x', 's']
    color_list = ['r', 'b', 'g']
    cmap = ListedColormap(color_list[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    t1 = np.linspace(x1_min, x1_max, 666)
    t2 = np.linspace(x2_min, x2_max, 666)

    x1, x2 = np.meshgrid(t1, t2)
    y_hat = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T)
    y_hat = y_hat.reshape(x1.shape)
    plt.contourf(x1, x2, y_hat, alpha=0.2, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    for ind, clas in enumerate(np.unique(y)):
        plt.scatter(X[y == clas, 0], X[y == clas, 1], alpha=0.8, s=50,
                    c=color_list[ind], marker=marker_list[ind], label=label_list[clas])


tree = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
tree.fit(X, y)

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=1,
            splitter='best')


plot_decision_regions(X, y, classifier=tree)
plt.xlabel('flower_Length(cm)', fontproperties='/Library/Fonts/Heiti.ttc')
plt.ylabel('flower_Width(cm)', fontproperties='/Library/Fonts/Heiti.ttc')
#plt.legend(prop=font)
plt.show()