import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import datasets, metrics
from collections import Counter
import matplotlib.pyplot as plt
import time
np.random.seed()
 
 
def prepare_data():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    return x, y

def eva_dbscan(x, y):
    k_begin = time.time()
    dbs = DBSCAN(eps=0.46,min_samples=11).fit(x, y)
    k_end = time.time() - k_begin
    pre_label = dbs.labels_
    key = list(set(pre_label))  # 获取簇标签
    # 重新定义簇标签
    key_c = []
    for i in key:
        idex = np.where(pre_label == i)
        key_c.append(Counter(y[idex]).most_common(1)[0][0])
    for i in range(pre_label.shape[0]):
        idex = key.index(pre_label[i])
        pre_label[i] = key_c[idex]
    dbs_ei = metrics.accuracy_score(y, pre_label) * 100
    dbs_rt = k_end
    plt.scatter(x[:, 0], x[:, 1], c=dbs.labels_)
    plt.show()
    return dbs_ei, dbs_rt
 
 
if __name__ == "__main__":
    x, y = prepare_data()
    dbs_ei, dbs_rt = eva_dbscan(x, y)
    print('DBSCAN:','accuracy:',dbs_ei,'%')
    print("        run_time:",dbs_rt,'s')