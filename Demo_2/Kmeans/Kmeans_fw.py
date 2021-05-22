'''K-M实现鸢尾花分类（ML大作业）'''
# by EotStxTaB in 20.12

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
lris_df = datasets.load_iris()

x_axis = lris_df.data[:,0]
y_axis = lris_df.data[:,2]
 
model = KMeans(n_clusters = 3)
 
# 训练模型
model.fit(lris_df.data)
 
#选取数据，进行预测
prddicted_label= model.predict([[6.3, 3.3, 6, 2.5]])
 
#预测全部150条数据
all_predictions = model.predict(lris_df.data)
 
#打印出来对150条数据的聚类散点图
plt.scatter(x_axis, y_axis, c = all_predictions)
plt.show()