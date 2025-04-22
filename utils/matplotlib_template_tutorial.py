import matplotlib.pyplot as plt
import numpy as np

'''
#*************************************************************************
#1.单一图形绘制示范：
x = [0, 1, 2, 3, 4]
y1 = [0, 1, 4, 9, 16]
y2 = [3, 6, 4, 7, 1]
plt.figure(figsize=(4,3),dpi=150)
# plt.figure命令非必需，是用于控制图形的相关属性。
# 但如果在此处设置了figsize的值，则将会影响到保存图片的尺寸
plt.plot(x, y1, label='Data Series1',
         color='blue',
         linestyle='-',
         marker='o')#绘制图形
plt.plot(x, y2, label='Data Series2',
color='red',
linestyle='--',
marker='o')#绘制图形
plt.xlabel("x",fontsize=15)
plt.ylabel("y",fontsize=15)
plt.xticks(fontsize=12)#设置x轴坐标文字的大小
plt.yticks(fontsize=15)
plt.title("title",fontsize=20,color="red")#设置图的标题
plt.grid(True,linestyle='--',color='grey',alpha=0.5)#设置网格
plt.legend(fontsize=15)#设置图例
plt.annotate('Important Point\nThis is a multi-line annotation',
             xy=(2, 4),
             xytext=(1, 8),
             arrowprops=dict(shrink=0.05, facecolor='blue', edgecolor='#fff111', width=2, headwidth=8, headlength=10,connectionstyle='arc3,rad=0.2'),
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
plt.savefig('figure.pdf',dpi=300, bbox_inches='tight')#必须在show()方法前调用才能实现图片的保存。bbox_inches='tight'确保图表内容不会被裁剪，完整保存
plt.show()#显示图片

#*************************************************************************
#2.多图形绘制示范：（详见9.网格图绘制）
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, y)
ax2.scatter(x, y)
plt.tight_layout()#用于自动调整子图的间距，避免图形重叠，应在所有子图绘制之后调用
plt.show()

#*************************************************************************
#3.堆叠图绘制示范：
N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, menMeans, width)
p2 = plt.bar(ind, womenMeans, width, bottom=menMeans)
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.show()

#*************************************************************************
#4.热图🔥绘制示范：
data = np.random.rand(10, 10)
heatmap = plt.imshow(data, cmap='viridis')
plt.colorbar(heatmap)
plt.show()

#*************************************************************************
#5.绘制3D scatter
from mpl_toolkits.mplot3d import Axes3D
x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

#*************************************************************************
#6.绘制3D 曲面图
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# 示例数据
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

#*************************************************************************
#7.绘制箱线图
# 两组学生的考试成绩数据
scores_group1 = [72, 85, 90, 67, 88, 75, 92, 84, 78, 65, 89, 95, 70, 82, 86, 76, 80, 91, 74, 87]
scores_group2 = [68, 78, 82, 70, 85, 76, 88, 74, 80, 69, 83, 90, 72, 81, 84, 75, 79, 87, 73, 86]# 绘制箱线图
# 将两组数据组合成一个列表
data = [scores_group1, scores_group2]
# 绘制箱线图
plt.boxplot(data)
# 添加标题和标签
plt.title('两组学生考试成绩分布')
plt.ylabel('成绩')
plt.xticks([1, 2], ['组1', '组2'])  # 设置x轴刻度标签
# 显示图形
plt.show()

#*************************************************************************
#8.绘制瀑布图
# 每月的收入变化（正数表示增加，负数表示减少）
changes = [100, -20, 50, -30, 80, -40, 70, -10, 60, -20, 90, -50]
# 月份
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# 计算累积值
cumulative = np.cumsum(changes)
# 创建一个新的图形
fig, ax = plt.subplots()
# 绘制柱状图
ax.bar(months, changes, color=['green' if x > 0 else 'red' for x in changes])
# 绘制累积值的折线图
ax.plot(months, cumulative, color='blue', marker='o')
# 添加标题和标签
ax.set_title('每月收入变化')
ax.set_xlabel('月份')
ax.set_ylabel('收入变化')
# 显示图形
plt.show()

#*************************************************************************
#9.网格图绘制
# 生成一些随机数据
np.random.seed(0)
data = np.random.rand(4, 10)  # 4个变量，每个变量10个数据点
# 创建一个 2x2 的网格图
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 在每个子图中绘制不同的变量
axs[0, 0].plot(data[0], label='Variable 1')
axs[0, 0].set_title('Variable 1')
axs[0, 0].legend()

axs[0, 1].plot(data[1], label='Variable 2')
axs[0, 1].set_title('Variable 2')
axs[0, 1].legend()

axs[1, 0].plot(data[2], label='Variable 3')
axs[1, 0].set_title('Variable 3')
axs[1, 0].legend()

axs[1, 1].plot(data[3], label='Variable 4')
axs[1, 1].set_title('Variable 4')
axs[1, 1].legend()

# 调整子图之间的间距
plt.tight_layout()
# 显示图形
plt.show()

#*************************************************************************
#10.直方图绘制
# 生成两组学生的考试成绩数据
scores_group1 = np.random.normal(75, 10, 1000)  # 组1
scores_group2 = np.random.normal(80, 10, 1000)  # 组2
# 绘制两组数据的直方图
plt.hist(scores_group1, bins=30, edgecolor='black', alpha=0.5, label='组1')
plt.hist(scores_group2, bins=30, edgecolor='black', alpha=0.5, label='组2')

# 添加标题和标签
plt.title('两组学生考试成绩分布')
plt.xlabel('成绩')
plt.ylabel('学生人数')
plt.legend()

# 显示图形
plt.show()

#*************************************************************************
# 11.颜色映射（Colormap）增强图表的视觉效果
#*************************************************************************
# 12.误差条（Error Bars）可以帮助展示数据的不确定性
# 两组实验数据
time_points = ['T1', 'T2', 'T3', 'T4']
means_group1 = [10, 15, 7, 5]  # 组1的均值
std_devs_group1 = [1, 2, 1.5, 0.5]  # 组1的标准差
means_group2 = [8, 12, 6, 4]  # 组2的均值
std_devs_group2 = [0.8, 1.8, 1.2, 0.4]  # 组2的标准差
# 创建折线图
plt.errorbar(time_points, means_group1, yerr=std_devs_group1, capsize=5, label='组1', marker='o', linestyle='-')
plt.errorbar(time_points, means_group2, yerr=std_devs_group2, capsize=5, label='组2', marker='s', linestyle='-')
# 添加标题和标签
plt.title('不同时间点的测量结果')
plt.xlabel('时间点')
plt.ylabel('测量值')
plt.legend()
# 显示图形
plt.show()

# *************************************************************************
# 13.使用对数轴（Log Scale）处理数据范围较大的情况
x = np.linspace(0.1, 10, 100)
y = np.exp(x)

plt.plot(x, y)
plt.yscale('log')  # 设置Y轴为对数尺度
plt.show()
'''

# *************************************************************************
# 14.性能对比分析（多柱状图）具体解释需将整段代码复制给kimi解释
# 示例数据
datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3']
performance_A = [80, 85, 90]
performance_B = [75, 78, 82]

x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, performance_A, width, label='Algorithm A')
rects2 = ax.bar(x + width/2, performance_B, width, label='Algorithm B')

ax.set_ylabel('Performance (%)')
ax.set_title('Performance comparison between two algorithms')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
plt.show()
