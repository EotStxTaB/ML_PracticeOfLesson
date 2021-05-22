import os
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

# 可视化整颗决策树
# filled=Ture添加颜色，rounded增加边框圆角
# out_file=None直接把数据赋给dot_data，不产生中间文件.dot
dot_data = export_graphviz(tree, filled=True, rounded=True,
                           class_names=['山鸢尾', '杂色鸢尾', '维吉尼亚鸢尾'],
                           feature_names=['花瓣长度', '花瓣宽度'], out_file=None)
graph = graph_from_dot_data(dot_data)
if not os.path.exists('代码-决策树.png'):
    graph.write_png('代码-决策树.png')

# 等比例改变图片大小
def cut_img(img_path, new_width, new_height=None):
    img = Image.open(img_path)
    width, height = img.size
    if new_height is None:
        new_height = int(height * (new_width / width))
    new_img = img.resize((new_width, new_height), Image.ANTIALIAS)
    os.remove(img_path)
    new_img.save(img_path)
    new_img.close()


cut_img('代码-决策树.png', 500)

# 只是为了展示图片，没有其他作用
img = imageio.imread('代码-决策树.png')
plt.imshow(img)
plt.show()