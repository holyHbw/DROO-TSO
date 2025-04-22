import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_rate(rate_his, rolling_intv=50):

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
#    rolling_intv = 20

    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.show()

def plot_data(x_data, y_data, label, xlabel, ylabel, file_name="defult",
              loc=1,c='r',marker=','):
    plt.figure()
    fig, ax = plt.subplots()
    ax.grid(True, linestyle='--', c='grey', alpha=0.4)
    plt.plot(x_data, y_data,label=label,c=c,marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    # plt.show()

    plt.savefig("./simulate_result/"+file_name+".png",dpi=300, bbox_inches='tight')

def plot_data_pro(x_data, y_data, label, xlabel, ylabel, file_name="defult",
              loc=1,c='r',marker=',',needMesh=False,needDetial=False,xstart=0,xend=0,ystart=0,yend=0):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_data, y_data,label=label,c=c,marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    if needMesh:
        ax.grid(True, linestyle='--', c='grey', alpha=0.4)
    if needDetial:
        # 使用 inset_axes 函数创建小图像
        axins = inset_axes(ax, width='35%', height='35%', loc='center')
        # 绘制细节图像
        axins.plot(x_data, y_data, label='Detail Data',c=c)
        # 设置细节图像的范围
        axins.set_xlim(xstart, xend)
        axins.set_ylim(ystart, yend)

    # plt.show()
    plt.savefig("./simulate_result/"+file_name+".png",dpi=300, bbox_inches='tight')

# def plot_line_together(x_data, y_data, label, xlabel, ylabel, file_name="defult",
#                        last=False,zorder=1,loc=1,c='r',marker=',', alpha=1):
#     # plt.figure(figsize=(8, 6))
#     plt.plot(x_data, y_data,label=label,zorder=zorder,color=c,marker=marker,markerfacecolor='none', alpha=alpha)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend(loc=loc)
#     if last:
#         plt.savefig("./simulate_result/" + file_name + ".png",dpi=300, bbox_inches='tight')

def plot_line_together(x_data, y_data, label, xlabel, ylabel, file_name="defult",
                       last=False,zorder=1,loc=1,c='r',marker=',', alpha=1):
    plt.plot(x_data, y_data,label=label,zorder=zorder,color=c,marker=marker,markerfacecolor='none', alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    if last:
        plt.savefig("./simulate_result/" + file_name + ".png",dpi=300, bbox_inches='tight')

# zorder 表示涂层显示的先后顺序，值越小则显示在越上层
# loc 表示 legend 显示位置（默认为1，即右上角）
# 通过 markerfacecolor 来指定这些标记内部的颜色
# label 用于描述 legend 中的内容
def plot_line_with_ax(ax, x_data, y_data,
                      label="default_label", xlabel="default_xlabel", ylabel="default_ylabel",
                      filename='',
                      zorder=1,loc=1,
                      c='r',alpha=1,
                      marker=None, markerfacecolor='none', markersize=6,
                      linestyle='-',linewidth=1.5,
                      offset=None):
    ax.plot(x_data, y_data, label=label, color=c, marker=marker,
            markerfacecolor=markerfacecolor, linestyle=linestyle, linewidth=linewidth,
            markersize=markersize, alpha=alpha,zorder=zorder)
    if xlabel != "default_xlabel":
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel != "default_ylabel":
        ax.set_ylabel(ylabel, fontsize=12)
    if label != "default_label":
        ax.legend(loc=loc, fontsize=10)
    if filename != '':
        plt.savefig("./simulate_result/" + filename + ".png",
                    dpi=300, bbox_inches='tight')
        plt.close()

# 向某一个图中添加细节子图
def add_detail_subfigure(ax, x_data, y_data, label, xlabel, ylabel, file_name="defult",
                       last=False,zorder=1,loc=1,c='r',marker=',', alpha=1):
    # plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data,label=label,zorder=zorder,color=c,marker=marker,markerfacecolor='none', alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    if last:
        plt.savefig("./simulate_result/" + file_name + ".png",dpi=300, bbox_inches='tight')

def plot_scatter_together(x_data, y_data, label, xlabel, ylabel, file_name="defult",
                          last=False,zorder=1,loc=1):
    plt.scatter(x_data, y_data,label=label,c='r',s=20,zorder=zorder)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    if last:
        plt.savefig("./simulate_result/" + file_name + ".png",dpi=300, bbox_inches='tight')

def plot_scatter(x_data, y_data, label, xlabel, ylabel, file_name="defult", last=False,zorder=1,loc=1):
    plt.figure()
    plt.scatter(x_data, y_data,label=label,c='r',s=20,zorder=zorder)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    if last:
        plt.savefig("./simulate_result/" + file_name + ".png",dpi=300, bbox_inches='tight')

def plot_pie(x_data, labels, colors, expload, file_name="defult"):
    plt.figure()
    plt.pie(x_data,  # 指定绘图数据
            labels=labels,  # 为饼图添加标签说明
            colors=colors,
            explode=expload,
            autopct='%.2f%%',  # 格式化输出百分比
            )
    plt.savefig("./simulate_result/" + file_name + ".png",dpi=300, bbox_inches='tight')