from utils.file_io_util import *
from utils.plot_util  import *
from config import N,TOTAL_TIME,LEARNING_RATE,TRAINING_INTERVAL,BATCH_SIZE,MEMORY,CASE

SUB_FOLDER = f"N{N}-epoch{TOTAL_TIME}-lr{LEARNING_RATE}-tr{TRAINING_INTERVAL}-bs{BATCH_SIZE}-ms{MEMORY}/"

# ************************************************1.GET DATA FROM FILE**********************************************#
'''get list'''
TSO_delay_arr = openfile_getdata("./simulate_result/"+SUB_FOLDER+"TSO_delay_arr.txt")
binary_delay_arr = openfile_getdata("./simulate_result/"+SUB_FOLDER+"binary_delay_arr.txt")
alloff_delay_arr = openfile_getdata("./simulate_result/"+SUB_FOLDER+"alloff_delay_arr.txt")
alllocal_delay_arr = openfile_getdata("./simulate_result/"+SUB_FOLDER+"alllocal_delay_arr.txt")
DROO_TSO_process_time_arr = openfile_getdata("./simulate_result/"+SUB_FOLDER+"DROO_TSO_process_time_arr.txt")
BINARY_process_time_arr = openfile_getdata("./simulate_result/"+SUB_FOLDER+"BINARY_process_time_arr.txt")
cost_arr = openfile_getdata("./simulate_result/"+SUB_FOLDER+"cost_arr.txt")
obj_opt_arr = openfile_getdata("./simulate_result/"+SUB_FOLDER+"obj_opt_arr.txt")
# t0_opt_arr = openfile_getdata("./simulate_result/"+SUB_FOLDER+"t0_opt_arr.txt")
prediction_trends_mean = openfile_getdata("./simulate_result/"+SUB_FOLDER+"prediction_trends_mean.txt")
prediction_trends_mean_avg = openfile_getdata("./simulate_result/"+SUB_FOLDER+"prediction_trends_mean_avg.txt")
prediction_trends_std = openfile_getdata("./simulate_result/"+SUB_FOLDER+"prediction_trends_std.txt")
prediction_trends_std_avg = openfile_getdata("./simulate_result/"+SUB_FOLDER+"prediction_trends_std_avg.txt")
prediction_trends_range = openfile_getdata("./simulate_result/"+SUB_FOLDER+"prediction_trends_range.txt")
prediction_trends_range_avg = openfile_getdata("./simulate_result/"+SUB_FOLDER+"prediction_trends_range_avg.txt")
strategy = openfile_getdata("./simulate_result/"+SUB_FOLDER+"strategy.txt")
'''get value'''
training_interval = int(openfile_getdata("./simulate_result/"+SUB_FOLDER+"TRAINING_INTERVAL.txt")[0])
# n = int(openfile_getdata("./simulate_result/"+SUB_FOLDER+"n.txt")[0])
# sampling0 = openfile_getdata("./simulate_result/"+SUB_FOLDER+"sampling0.txt")[0]
# sampling1 = openfile_getdata("./simulate_result/"+SUB_FOLDER+"sampling1.txt")[0]
# tloc = openfile_getdata("./simulate_result/"+SUB_FOLDER+"tloc.txt")
# toff = openfile_getdata("./simulate_result/"+SUB_FOLDER+"toff.txt")
# E_total = openfile_getdata("./simulate_result/"+SUB_FOLDER+"E_total.txt")
# E_off = openfile_getdata("./simulate_result/"+SUB_FOLDER+"E_off.txt")
# E_loc = openfile_getdata("./simulate_result/"+SUB_FOLDER+"E_loc.txt")

# t0 = openfile_getdata("./simulate_result/"+SUB_FOLDER+"tt.txt")[0]
# Tdelay = openfile_getdata("./simulate_result/"+SUB_FOLDER+"delay1.txt")[0]

# ************************************************2.PLOT loss**********************************************#
plot_data_pro(np.arange(len(cost_arr)), cost_arr,
                  "Value of training Loss",
                  'Training Times',
                  'Training Loss',
                  SUB_FOLDER+"loss",c='b',needMesh=True,needDetial=True,xstart=len(cost_arr)*0.8,xend=len(cost_arr),ystart=cost_arr[int(len(cost_arr)*0.8)]/4,yend=cost_arr[int(len(cost_arr)*0.8)]*2)

# ************************************************3.PLOT TCD separately**********************************************#
if CASE==1:
    new_TSO_delay_arr = []
    new_binary_delay_arr = []
    new_alloff_delay_arr = []
    new_alllocal_delay_arr = []
    gap = 20
    for i in range(0, len(TSO_delay_arr)):
        if i % gap == 0:
            new_TSO_delay_arr.append(TSO_delay_arr[i])
            new_binary_delay_arr.append(binary_delay_arr[i])
            new_alloff_delay_arr.append(alloff_delay_arr[i])
            new_alllocal_delay_arr.append(alllocal_delay_arr[i])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, linestyle='--', c='grey', alpha=0.4)

    plot_line_with_ax(ax, x_data=np.arange(len(new_TSO_delay_arr)), y_data=new_TSO_delay_arr,
                      label="TCD with TSO MODE", xlabel=f'Time(every {gap * TRAINING_INTERVAL} frames)',
                      ylabel="TCD(seconds)", c='r', marker='v')
    plot_line_with_ax(ax, np.arange(len(new_binary_delay_arr)), new_binary_delay_arr,
                      "TCD with Binary Offloading MODE", xlabel=f'Time(every {gap * TRAINING_INTERVAL} frames)',
                      ylabel="TCD(seconds)",
                      c='y', marker='*')
    plot_line_with_ax(ax, np.arange(len(new_alloff_delay_arr)), new_alloff_delay_arr,
                      "TCD with AO MODE", xlabel=f'Time(every {gap * TRAINING_INTERVAL} frames)',
                      ylabel="TCD(seconds)",
                      c='c', marker='s')
    plot_line_with_ax(ax, np.arange(len(new_alllocal_delay_arr)), new_alllocal_delay_arr,
                      "TCD with ALC MODE", xlabel=f'Time(every {gap * TRAINING_INTERVAL} frames)',
                      ylabel="TCD(seconds)",
                      c='b', marker='o')

    # 添加细节图
    # ----------------------------------------------
    # 设置细节区域参数
    detail_start = int(len(new_TSO_delay_arr) * 0.8)  # 取最后20%数据
    detail_end = len(new_TSO_delay_arr)
    detail_ylim1 = (min(min(arr[-10:]) for arr in [new_TSO_delay_arr,
                                                   new_binary_delay_arr,
                                                   new_alllocal_delay_arr,new_alloff_delay_arr]),
                    max(max(arr[-10:]) for arr in [new_TSO_delay_arr,
                                                   new_binary_delay_arr,
                                                   new_alllocal_delay_arr,new_alloff_delay_arr]))

    # 创建细节子图， 在ax中嵌入一个子图
    axins1 = inset_axes(ax, width="35%", height="35%", loc='center right',
                        borderpad=1.5)
    axins1.grid(True, linestyle=':', c='grey', alpha=0.3)

    # 设置细节图范围
    axins1.set_xlim(detail_start - 0.5, detail_end - 0.5)  # 优化显示范围
    axins1.set_ylim(detail_ylim1[0] * 0.98, detail_ylim1[1] * 1.02)

    # 添加细节框指示
    ax.indicate_inset_zoom(axins1, edgecolor="black", alpha=0.5)

    # 保持与主图一致的样式
    plot_line_with_ax(axins1, np.arange(detail_start, detail_end),
                      new_TSO_delay_arr[detail_start:detail_end],
                      c='r', marker='v', alpha=0.8)
    plot_line_with_ax(axins1, np.arange(detail_start, detail_end),
                      new_binary_delay_arr[detail_start:detail_end],
                      c='y', marker='*', alpha=0.8)
    plot_line_with_ax(axins1, np.arange(detail_start, detail_end),
                      new_alloff_delay_arr[detail_start:detail_end],
                      c='c', marker='s', alpha=0.8)
    plot_line_with_ax(axins1, np.arange(detail_start, detail_end),
                      new_alllocal_delay_arr[detail_start:detail_end],
                      c='b', marker='o', alpha=0.8, filename=f"{SUB_FOLDER}TCD_of_4mode_with_detail")
    # ----------------------------------------------

if CASE==2:
    new_TSO_delay_arr = []
    new_binary_delay_arr = []
    new_alloff_delay_arr = []
    new_alllocal_delay_arr = []
    gap = 20
    for i in range(0, len(TSO_delay_arr)):
        if i % gap == 0:
            new_TSO_delay_arr.append(TSO_delay_arr[i])
            new_binary_delay_arr.append(binary_delay_arr[i])
            new_alloff_delay_arr.append(alloff_delay_arr[i])
            new_alllocal_delay_arr.append(alllocal_delay_arr[i])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, linestyle='--', c='grey', alpha=0.4)

    plot_line_with_ax(ax, x_data=np.arange(len(new_TSO_delay_arr)), y_data=new_TSO_delay_arr,
                            label="TCD with TSO MODE", xlabel=f'Time(every {gap * TRAINING_INTERVAL} frames)',
                            ylabel="TCD(seconds)", c='r', marker='v')
    plot_line_with_ax(ax, np.arange(len(new_binary_delay_arr)), new_binary_delay_arr,
                            "TCD with Binary Offloading MODE", xlabel=f'Time(every {gap * TRAINING_INTERVAL} frames)',
                            ylabel="TCD(seconds)",
                            c='y', marker='*')
    plot_line_with_ax(ax, np.arange(len(new_alloff_delay_arr)), new_alloff_delay_arr,
                            "TCD with AO MODE", xlabel=f'Time(every {gap * TRAINING_INTERVAL} frames)',
                            ylabel="TCD(seconds)",
                            c='c', marker='s')
    plot_line_with_ax(ax, np.arange(len(new_alllocal_delay_arr)), new_alllocal_delay_arr,
                            "TCD with ALC MODE", xlabel=f'Time(every {gap * TRAINING_INTERVAL} frames)',
                            ylabel="TCD(seconds)",
                            c='b', marker='o')

    # 添加细节图
    # ----------------------------------------------
    # 设置细节区域参数
    detail_start = int(len(new_TSO_delay_arr) * 0.8)  # 取最后20%数据
    detail_end = len(new_TSO_delay_arr)
    detail_ylim1 = (min(min(arr[-10:]) for arr in [new_TSO_delay_arr,
                                                   new_binary_delay_arr,
                                                   new_alllocal_delay_arr]),
                    max(max(arr[-10:]) for arr in [new_TSO_delay_arr,
                                                   new_binary_delay_arr,
                                                   new_alllocal_delay_arr]))

    # 创建细节子图， 在ax中嵌入一个子图
    axins1 = inset_axes(ax, width="35%", height="35%", loc='center right',
                        borderpad=1.5)
    axins1.grid(True, linestyle=':', c='grey', alpha=0.3)

    # 设置细节图范围
    axins1.set_xlim(detail_start - 0.5, detail_end - 0.5)  # 优化显示范围
    axins1.set_ylim(detail_ylim1[0] * 0.98, detail_ylim1[1] * 1.02)

    # 添加细节框指示
    ax.indicate_inset_zoom(axins1, edgecolor="black", alpha=0.5)

    # 保持与主图一致的样式
    plot_line_with_ax(axins1, np.arange(detail_start, detail_end),
                            new_TSO_delay_arr[detail_start:detail_end],
                            c='r', marker='v', alpha=0.8)
    plot_line_with_ax(axins1, np.arange(detail_start, detail_end),
                            new_binary_delay_arr[detail_start:detail_end],
                            c='y', marker='*', alpha=0.8)
    plot_line_with_ax(axins1, np.arange(detail_start, detail_end),
                            new_alllocal_delay_arr[detail_start:detail_end],
                            c='b', marker='o', alpha=0.8, filename=f"{SUB_FOLDER}TCD_of_4mode_with_detail")
    # ----------------------------------------------

if CASE==3:
    new_TSO_delay_arr = []
    new_binary_delay_arr = []
    new_alloff_delay_arr = []
    new_alllocal_delay_arr = []
    gap = 20
    for i in range(0, len(TSO_delay_arr)):
        if i % gap == 0:
            new_TSO_delay_arr.append(TSO_delay_arr[i])
            new_binary_delay_arr.append(binary_delay_arr[i])
            new_alloff_delay_arr.append(alloff_delay_arr[i])
            new_alllocal_delay_arr.append(alllocal_delay_arr[i])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, linestyle='--', c='grey', alpha=0.4)

    # 统一绘图函数调用（移除保存操作）
    def plot_line_together_temp(ax, x_data, y_data, label, xlabel, ylabel,
                           c='r', marker='o', alpha=1):
        ax.plot(x_data, y_data, label=label, color=c, marker=marker,
                markerfacecolor='none', linestyle='-', linewidth=1.5,
                markersize=6, alpha=alpha)

    # 主图绘制
    plot_line_together_temp(ax, np.arange(len(new_TSO_delay_arr)), new_TSO_delay_arr,
                       "TCD with TSO MODE", f'Time(every {gap * TRAINING_INTERVAL} frames)',
                       "TCD", c='r', marker='v')
    plot_line_together_temp(ax, np.arange(len(new_binary_delay_arr)), new_binary_delay_arr,
                       "TCD with Binary Offloading MODE", "", "",
                       c='y', marker='*')
    plot_line_together_temp(ax, np.arange(len(new_alloff_delay_arr)), new_alloff_delay_arr,
                       "TCD with AO MODE", "", "",
                       c='c', marker='s')
    plot_line_together_temp(ax, np.arange(len(new_alllocal_delay_arr)), new_alllocal_delay_arr,
                       "TCD with ALC MODE", "", "",
                       c='b', marker='o')

    # 添加细节图1
    # ----------------------------------------------
    # 设置细节区域参数
    detail_start = int(len(new_TSO_delay_arr) * 0.8)  # 取最后20%数据
    detail_end = len(new_TSO_delay_arr)
    detail_ylim1 = (min(min(arr[-10:]) for arr in [new_TSO_delay_arr,
                                                  new_binary_delay_arr,
                                                  new_alloff_delay_arr]),
                   max(max(arr[-10:]) for arr in [new_TSO_delay_arr,
                                                  new_binary_delay_arr,
                                                  new_alloff_delay_arr]))

    # 创建细节子图
    axins1 = inset_axes(ax, width="25%", height="35%", loc=10,
                       borderpad=1.5)
    axins1.grid(True, linestyle=':', c='grey', alpha=0.3)

    # 保持与主图一致的样式
    plot_line_together_temp(axins1, np.arange(detail_start, detail_end),
                       new_TSO_delay_arr[detail_start:detail_end],
                       "", "", "", c='r', marker='v', alpha=0.8)
    plot_line_together_temp(axins1, np.arange(detail_start, detail_end),
                       new_binary_delay_arr[detail_start:detail_end],
                       "", "", "", c='y', marker='*', alpha=0.8)
    plot_line_together_temp(axins1, np.arange(detail_start, detail_end),
                       new_alloff_delay_arr[detail_start:detail_end],
                       "", "", "", c='c', marker='s', alpha=0.8)

    # 设置细节图范围
    axins1.set_xlim(detail_start - 0.5, detail_end - 0.5)  # 优化显示范围
    axins1.set_ylim(detail_ylim1[0] * 0.98, detail_ylim1[1] * 1.02)

    # 添加细节框指示
    ax.indicate_inset_zoom(axins1, edgecolor="black", alpha=0.5)
    # ----------------------------------------------

    # 添加细节图2
    # ----------------------------------------------
    # 设置细节区域参数
    detail_start = int(len(new_alllocal_delay_arr) * 0.8)  # 取最后20%数据
    detail_end = len(new_alllocal_delay_arr)
    detail_ylim2 = (min(min(arr[-10:]) for arr in [new_alllocal_delay_arr]),
                    max(max(arr[-10:]) for arr in [new_alllocal_delay_arr]))

    # 创建细节子图
    axins2 = inset_axes(ax, width="25%", height="35%", loc='center right',
                       borderpad=1.5)
    axins2.grid(True, linestyle=':', c='grey', alpha=0.3)

    # 保持与主图一致的样式
    plot_line_together_temp(axins2, np.arange(detail_start, detail_end),
                            new_alllocal_delay_arr[detail_start:detail_end],
                            "", "", "", c='b', marker='o', alpha=0.8)

    # 设置细节图范围
    axins2.set_xlim(detail_start - 0.5, detail_end - 0.5)  # 优化显示范围
    axins2.set_ylim(detail_ylim2[0] * 0.98, detail_ylim2[1] * 1.02)

    # 添加细节框指示
    ax.indicate_inset_zoom(axins2, edgecolor="black", alpha=0.5)
    # ----------------------------------------------

    # 统一设置图例和标签
    ax.set_xlabel(f'Time(every {gap * TRAINING_INTERVAL} frames)', fontsize=12)
    ax.set_ylabel("TCD", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    # 最终保存
    plt.savefig(f"./simulate_result/{SUB_FOLDER}TCD_of_4mode_with_detail.png",
                dpi=300, bbox_inches='tight')
    plt.close()

# ************************************************4.PLOT algo's process time comparation**********************************************#
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(True, linestyle='--', c='grey', alpha=0.4)
plot_line_with_ax(ax, x_data=np.arange(len(DROO_TSO_process_time_arr)), y_data=DROO_TSO_process_time_arr,
                   label="Processing Time of DROO-TSO",
                   xlabel=f'Time(every {TRAINING_INTERVAL} frames)',
                   ylabel="Processing Time for Different Algorithms (in seconds)",c='r')
plot_line_with_ax(ax, np.arange(len(BINARY_process_time_arr)), BINARY_process_time_arr,
                   "Processing Time of Binary-Offloading",
                   f'Time(every {TRAINING_INTERVAL} frames)',
                   "Processing Time for Different Algorithms (in seconds)",c='c')
# 添加细节图
# 设置细节区域参数
detail_start = int(len(DROO_TSO_process_time_arr) * 0.8)  # 取最后20%数据
detail_end = len(DROO_TSO_process_time_arr)
detail_ylim1 = (min(min(arr[-10:]) for arr in [DROO_TSO_process_time_arr]),
                max(max(arr[-10:]) for arr in [DROO_TSO_process_time_arr]))

# 创建细节子图， 在ax中嵌入一个子图
axins1 = inset_axes(ax, width="25%", height="25%", loc='center right',
                    borderpad=1.5,
                    bbox_to_anchor=(0, -0.2, 1, 1),  # 这里的0和-0.2是相对于当前子图所处的'center right'为坐标原点
                    bbox_transform=ax.transAxes  )# 指定相对于父轴的比例
axins1.grid(True, linestyle=':', c='grey', alpha=0.3)

# 设置细节图范围
axins1.set_xlim(detail_start - 0.5, detail_end - 0.5)  # 优化显示范围
axins1.set_ylim(detail_ylim1[0] * 0.98, detail_ylim1[1] * 1.02)

# 添加细节框指示
# ax.indicate_inset_zoom(axins1, edgecolor="black", alpha=0.5)

# 保持与主图一致的样式
plot_line_with_ax(axins1, np.arange(detail_start, detail_end),
                        DROO_TSO_process_time_arr[detail_start:detail_end],
                        c='r', alpha=0.8,filename=f"{SUB_FOLDER}algo_process_time_comparation")

# ************************************************5.PLOT percentage**********************************************#
# plot_pie([sampling0,sampling1],['recharge time','computing time'],['g','r'],[0,0],SUB_FOLDER+'percentage')

# ************************************************6.PLOT strategy**********************************************#
fig,ax = plt.subplots(figsize=(10, 6))
wd = ['WD1', 'WD2', 'WD3', 'WD4', 'WD5', 'WD6', 'WD7', 'WD8', 'WD9', 'WD10']
strategies_loc = []
for i in strategy:
    strategies_loc.append(1-i)
bar_width = 0.6
index1 = np.arange(len(wd))
bars1=plt.bar(index1, height=strategy,width=bar_width, color='gold', label='Offload Percentage')
bars2=plt.bar(index1, height=strategies_loc, bottom=strategy,width=bar_width, color='tomato', label='Local Computing Percentage')

# 添加文本标签
for bar1, bar2, s, sl in zip(bars1, bars2, strategy, strategies_loc):
    # 在 offload 部分添加文本
    height1 = bar1.get_height()
    ax.text(bar1.get_x() + bar1.get_width() / 2., height1 / 2, f'{s:.0%}',
            ha='center', va='center', color='black', fontsize=9)

    # 在 local computing 部分添加文本
    height2 = bar2.get_height()
    total_height = bar1.get_height() + bar2.get_height()
    ax.text(bar2.get_x() + bar2.get_width() / 2., total_height - height2 / 2, f'{sl:.0%}',
            ha='center', va='center', color='black', fontsize=9)

plt.legend(loc=1,borderpad=0.1,bbox_to_anchor=(1.01,1))  # 显示图例
# plt.xticks(index1 + bar_width/3, BD3list,rotation=90)
# plt.ylim(0,0.6)
plt.ylabel('The percentage of offload and local computing')  # 纵坐标轴标题
plt.xlabel('WDs')
plt.savefig("./simulate_result/" + SUB_FOLDER + "strategy.png", dpi=300, bbox_inches='tight')

# ************************************************7.PLOT time**********************************************#
# plt.figure()
# wd = ['t0','t_loc_1', 't_loc_2', 't_loc_3', 't_loc_4', 't_loc_5', 't_loc_6', 't_loc_7', 't_loc_8', 't_loc_9', 't_loc_10','t_off_all']
# width = 1
# a=[0]
# b=[1,2,3,4,5,6,7,8,9,10]
# c=[11]
# plt.bar(a,t0,width=width,label='t0',color='g')
# plt.bar(b,tloc,width=width,label='t_loc',color='b')
# plt.bar(c,toff[0],width=width,label='t_off_wd1',color='r')
# plt.bar(c,toff[1],bottom=toff[0],width=width,label='t_off_wd2',color='r')
# plt.bar(c,toff[2],bottom=toff[0]+toff[1],width=width,label='t_off_wd3',color='r')
# plt.bar(c,toff[3],bottom=toff[0]+toff[1]+toff[2],width=width,label='t_off_wd34',color='r')
# plt.bar(c,toff[4],bottom=toff[0]+toff[1]+toff[2]+toff[3],width=width,label='t_off_wd5',color='r')
# plt.bar(c,toff[5],bottom=toff[0]+toff[1]+toff[2]+toff[3]+toff[4],width=width,label='t_off_wd6',color='r')
# plt.bar(c,toff[6],bottom=toff[0]+toff[1]+toff[2]+toff[3]+toff[4]+toff[5],width=width,label='t_off_wd7',color='r')
# plt.bar(c,toff[7],bottom=toff[0]+toff[1]+toff[2]+toff[3]+toff[4]+toff[5]+toff[6],width=width,label='t_off_wd8',color='r')
# plt.bar(c,toff[8],bottom=toff[0]+toff[1]+toff[2]+toff[3]+toff[4]+toff[5]+toff[6]+toff[7],width=width,label='t_off_wd9',color='r')
# plt.bar(c,toff[9],bottom=toff[0]+toff[1]+toff[2]+toff[3]+toff[4]+toff[5]+toff[6]+toff[7]+toff[8],width=width,label='t_off_wd10',color='r')
# plt.xlabel('1')
# plt.ylabel('2')
# plt.legend()
# plt.savefig("./simulate_result/" + SUB_FOLDER + "time_detial.png")

# ************************************************8.PLOT energy**********************************************#
# plt.figure()
# a = [0,2,4,6,8,10,12,14,16,18]
# b = [1,3,5,7,9,11,13,15,17,19]
# width = 1
# plt.bar(a,E_total,width=width,label='E_total',color='g')
# plt.bar(b,E_off,width=width,label='E_off',color='r')
# plt.bar(b,E_loc,bottom=E_off,width=width,label='E_loc',color='y')
# plt.xlabel('3')
# plt.ylabel('4')
# plt.legend()
# plt.savefig("./simulate_result/" + SUB_FOLDER + "energy_detial.png")

# ************************************************9.PLOT network mode**********************************************#
# from torchviz import make_dot
#
# LEARNING_RATE = 0.0001
# TRAINING_INTERVAL = 20
# BATCH_SIZE = 64
# Memory = 1024                # capacity of memory structure
# decoder_mode = 'DELTA_KNN'   # the quantization mode could be 'KNM' or 'KNN' or 'MT_KNN' or 'DELTA_KNN'
# Delta = 32
# mem = MemoryDNN(net = [10, 120, 80, 10],
#                     learning_rate = LEARNING_RATE,
#                     training_interval=TRAINING_INTERVAL,
#                     batch_size=BATCH_SIZE,
#                     memory_size=Memory)
# make_dot(strategy, params=dict(list(mem.named_parameters()))).render("torchviz", format="png")

# ************************************************10.Prediction trends**********************************************#
from config import PREDICTION_AVG_GAP
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(True, linestyle='--',c='grey',alpha=0.4)
x_values = np.arange(PREDICTION_AVG_GAP, TOTAL_TIME, PREDICTION_AVG_GAP)
plot_line_together(np.arange(len(prediction_trends_mean)), prediction_trends_mean,
                   "The mean of the model's predicted values",
                   'Time',
                   "TCD",loc='center right',c='r',alpha=0.3)
plot_line_together(x_values, prediction_trends_mean_avg,
                   "50-point moving average of the mean value",
                   'Time',
                   "TCD",loc='center right',c='r')
plot_line_together(np.arange(len(prediction_trends_std)), prediction_trends_std,
                   "The standard deviation of the model's predicted values",
                   'Time',
                   "TCD",loc='center right',c='y',alpha=0.3)
plot_line_together(x_values, prediction_trends_std_avg,
                   "50-point moving average of the standard deviation value",
                   'Time',
                   "TCD",loc='center right',c='y')
plot_line_together(np.arange(len(prediction_trends_range)), prediction_trends_range,
                   "The range of the model's predicted values",
                   'Time',
                   "Trends", loc='center right',c='b',alpha=0.3)
plot_line_together(x_values, prediction_trends_range_avg,
                   "50-point moving average of the range value",
                   'Time',
                   "Trends", SUB_FOLDER + "Prediction_trends",True,loc='center right',c='b')

