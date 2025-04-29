import numpy as np
import scipy.io as sio
import math

#******************************* Parameters for log **********************************************
LOG_ON_MAIN = False
LOG_ON_OPTIMIZATION = False
CVX_ON = False

#******************************* Parameters for system model *************************************
# We consider 3 situations:
#    case 1: Local and Offload are on equal footing
#    case 2: Local better than Offload
#    case 3: Offload better than Local
CASE = 1

if CASE == 1:
    #******************************* - 1.Local and Offload are on equal footing ************************
    N = 10 # number of users
    AP_POWER_dBm = 50.0  #AP的发射功率，单位是分贝毫瓦（dBm）。若要转换为瓦（W），则=10^((dBm-30)/10)
    AP_POWER_W = 10 ** ((AP_POWER_dBm-30)/10)  #AP的发射功率，单位是瓦（W）。
    P_WD_MAX_dBm = 23.0 #节点的发射功率，单位是分贝毫瓦（dBm）。若要转换为瓦（W），则=10^((dBm-30)/10)
    P_WD_MAX_W = 10 ** ((P_WD_MAX_dBm-30)/10)  #节点的发射功率，单位是瓦（W）
    F_WD_MAX = 3e7 #本地计算的cpu处理频率，次/秒
    RECHARGE_EFFICIENCY = 0.85#*10**6  # recharge efficiency
    BANDWIDTH = 1e6  # 带宽，单位要转换为Hz，1MHz就是1e6
    XIGEMA_2_dBm = 10 * math.log10(1.38e-23 * 290 * BANDWIDTH / 0.001)  #表示接收端噪声功率，单位为分贝毫瓦（dBm）
    XIGEMA_2_W = 10 ** ((XIGEMA_2_dBm-30)/10)  #表示接收端噪声功率，单位为瓦（W）
    EPSILON = np.ones(N) * 1e-9 # cpu energy efficiency，单位是焦耳·秒/赫兹³
    G = np.ones(N) * 1e-4 # 每处理1bit数据所需的时钟周期数
    '''关于L，需要每个epoch都生成不同的数据，还是所有epoch都采用同一组数据？需要思考。 暂时先使用同一组数据'''
    L = np.random.uniform(0.5, 2, size=N) * 1e6  # bits for every WD, 0.5~2Mbit，单位为bit
elif CASE == 2:
    #******************************* - 2.Local better than Offload *************************************
    N = 10 # number of users
    AP_POWER_dBm = 50.0  #AP的发射功率，单位是分贝毫瓦（dBm）。若要转换为瓦（W），则=10^((dBm-30)/10)
    AP_POWER_W = 10 ** ((AP_POWER_dBm-30)/10)  #AP的发射功率，单位是瓦（W）。
    P_WD_MAX_dBm = 23.0 #节点的发射功率，单位是分贝毫瓦（dBm）。若要转换为瓦（W），则=10^((dBm-30)/10)
    P_WD_MAX_W = 10 ** ((P_WD_MAX_dBm-30)/10)  #节点的发射功率，单位是瓦（W）
    F_WD_MAX = 3e7 #次/秒
    RECHARGE_EFFICIENCY = 0.85#*10**6  # recharge efficiency
    BANDWIDTH = 1e6  # 单位要转换为Hz，1MHz就是1e6
    XIGEMA_2_dBm = 10 * math.log10(1.38e-23 * 290 * BANDWIDTH / 0.001)  #表示接收端噪声功率，单位为分贝毫瓦（dBm）
    XIGEMA_2_W = 10 ** ((XIGEMA_2_dBm-30)/10)  #表示接收端噪声功率，单位为瓦（W）
    EPSILON = np.ones(N) * 1e-9 # cpu energy efficiency，单位是焦耳·秒/赫兹³
    G = np.ones(N) * 1e-6
    '''关于L，需要每个epoch都生成不同的数据，还是所有epoch都采用同一组数据？需要思考。 暂时先使用同一组数据'''
    L = np.random.uniform(0.5, 2, size=N) * 1e6  # bits for every WD, 0.5~2Mbit，单位为bit
elif CASE == 3:
    #******************************* - 3.Offload better than Local *************************************
    N = 10 # number of users
    AP_POWER_dBm = 50.0  #AP的发射功率，单位是分贝毫瓦（dBm）。若要转换为瓦（W），则=10^((dBm-30)/10)
    AP_POWER_W = 10 ** ((AP_POWER_dBm-30)/10)  #AP的发射功率，单位是瓦（W）。
    P_WD_MAX_dBm = 23.0 #节点的发射功率，单位是分贝毫瓦（dBm）。若要转换为瓦（W），则=10^((dBm-30)/10)
    P_WD_MAX_W = 10 ** ((P_WD_MAX_dBm-30)/10)  #节点的发射功率，单位是瓦（W）
    F_WD_MAX = 3e7 #次/秒
    RECHARGE_EFFICIENCY = 0.85#*10**6  # recharge efficiency
    BANDWIDTH = 1e6  # 单位要转换为Hz，1MHz就是1e6
    XIGEMA_2_dBm = 10 * math.log10(1.38e-23 * 290 * BANDWIDTH / 0.001)  #表示接收端噪声功率，单位为分贝毫瓦（dBm）
    XIGEMA_2_W = 10 ** ((XIGEMA_2_dBm-30)/10)  #表示接收端噪声功率，单位为瓦（W）
    EPSILON = np.ones(N) * 1e-9 # cpu energy efficiency，单位是焦耳·秒/赫兹³
    G = np.ones(N) * 100
    '''关于L，需要每个epoch都生成不同的数据，还是所有epoch都采用同一组数据？需要思考。 暂时先使用同一组数据'''
    L = np.random.uniform(0.5, 2, size=N) * 1e6  # bits for every WD, 0.5~2Mbit，单位为bit
else:
    print("Wrong CASE!!!")

#******************************* Parameters for training DNN *************************************
TOTAL_TIME = 20000           # number of time frames
K = N                        # initialize K = N
DNN_SIZE = [N, 120, 80, N]
LEARNING_RATE = 0.0001 #0.0001
TRAINING_INTERVAL = 20
BATCH_SIZE = 64
MEMORY = 1024                # capacity of memory structure
DECODER_MODE = 'DELTA_KNN'   # the quantization mode could be 'KNM' or 'KNN' or 'MT_KNN' or 'DELTA_KNN'
DELTA = 32                   # Update interval for adaptive K
PREDICTION_AVG_GAP = 50

#******************************* Parameters for channel gain *************************************
CHANNEL_GAINS_ORIGIN = 10 ** (-np.random.uniform(30, 40, size=(TOTAL_TIME*2, N))/10) #-30dB～-40dB, 0~?m
# set CHANNEL_GAINS close to 1 for better training
# Step 1: 对数变换（压缩动态范围）
log_h = np.log10(CHANNEL_GAINS_ORIGIN)
# Step 2: 标准化（转换为零均值、单位方差）
log_h_mean = log_h.mean()
log_h_std = log_h.std()
CHANNEL_GAINS_STANDARDIZED = (log_h - log_h_mean) / log_h_std  #将CHANNEL_GAINS_ORIGIN进行标准化，作为模型的input

SPLIT_INDEX = int(.8 * len(CHANNEL_GAINS_STANDARDIZED)) #use 80% data to train

#******************************* Parameters for file path *************************************
PIC_PATH = "./simulate_result/"+"N" + str(N) \
               +"-epoch" + str(TOTAL_TIME) \
               +"-lr" + str(LEARNING_RATE) \
               +"-tr" + str(TRAINING_INTERVAL) \
               +"-bs" + str(BATCH_SIZE) \
               +"-ms" + str(MEMORY) + "/"

