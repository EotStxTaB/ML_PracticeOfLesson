# DecisionTreeClassifier参数含义
* criterion：gini或者entropy,前者是基尼系数，后者是信息熵。
* splitter： best or random 前者是在所有特征中找最好的切分点 后者是在部分特征中，默认的”best”适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐”random” 。
* max_features：None（所有），log2，sqrt，N  特征小于50的时候一般使用所有的
* max_depth：  int or None, optional (default=None) 设置决策随机森林中的决策树的最大深度，深度越大，越容易过拟合，推荐树的深度为：5-20之间。
* min_samples_split：设置结点的最小样本数量，当样本数量可能小于此值时，结点将不会在划分。
* min_samples_leaf： 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。
* min_weight_fraction_leaf： 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝默认是0，就是不考虑权重问题。
* max_leaf_nodes： 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。
* class_weight： 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重，如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。
* min_impurity_split： 这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值则该节点不再生成子节点。即为叶子节点 。
