#  #################################################################
#  DROO-TSO Algo
#  version 2.0 -- Feb 2025. Written by Bowen Huang (3240009761@student.must.edu.mo)
#  #################################################################

from memory_dnn import MemoryDNN
# from memory_dnn import MemoryDNN
from optimization import *
from utils.plot_util import *
from utils.file_io_util import *
from utils.binary_policy_generator import generate_binary_policy
import time
from config import *
import faulthandler

if __name__ == "__main__":

    faulthandler.enable()

    if LOG_ON_MAIN:
        print('#user = %d, #TOTAL_TIME=%d, K=%d, decoder = %s, Memory = %d, Delta = %d, epsilon = %d, G = %d' % (N, TOTAL_TIME, K, DECODER_MODE, MEMORY, DELTA,EPSILON,G))

    mem = MemoryDNN(net = DNN_SIZE,
                    learning_rate = LEARNING_RATE,
                    training_interval=TRAINING_INTERVAL,
                    batch_size=BATCH_SIZE,
                    memory_size=MEMORY)

    #create path for result
    os.makedirs(PIC_PATH, exist_ok=True)

    # ************************************************FOR START**********************************************#
    t0_opt_arr = []
    obj_opt_arr = [] # 记录每个epoch下droo+offloading_first计算出的最优time delay
    obj_opt_arr_cvx = [] # 记录每个epoch下droo+cvx计算出的最优time delay
    tso_strategy_history = [] # 用于存储在每一次测试产生的TSO决策
    binary_strategy_history = [] # 用于存储在每一次测试产生的最佳二分决策
    prediction_trends_mean = []  # 用于记录每个epoch下的预测值的 均值，方便后续分析预测的变化趋势（主要是看预测是否向预定目标收敛）
    prediction_trends_mean_avg = []  # 用于记录prediction_trends_mean的均值变化
    prediction_trends_std = []   # 用于记录每个epoch下的预测值的 标准差，方便后续分析预测的变化趋势（主要是看预测是否向预定目标收敛）
    prediction_trends_std_avg = []   # prediction_trends_std
    prediction_trends_range = [] # 用于记录每个epoch下的预测值的 极差，方便后续分析预测的变化趋势（主要是看预测是否向预定目标收敛）
    prediction_trends_range_avg = [] # prediction_trends_range
    TSO_delay_arr = []
    binary_delay_arr = []
    alloff_delay_arr = []
    alllocal_delay_arr = []
    DROO_TSO_process_time_arr = []
    BINARY_process_time_arr = []

    for epoch in range(TOTAL_TIME):


        if (epoch+1)%1000 == 0:
            print('SLOT =',epoch+1)

        i_idx = epoch % SPLIT_INDEX
        h_standard = CHANNEL_GAINS_STANDARDIZED[i_idx,:] #input for this epoch
        h_origin = CHANNEL_GAINS_ORIGIN[i_idx,:]
        #************************************************DECODE**********************************************#
        sub_strategy = mem.decode(h_standard, epoch, K, N, DECODER_MODE)#生成了2*K+1个子决策（对于部分卸载而言）,这个决策指的是每个节点的卸载比例
        prediction_trends_mean.append(np.mean(sub_strategy[-1]))
        prediction_trends_std.append(np.std(sub_strategy[-1]))
        prediction_trends_range.append(np.max(sub_strategy[-1]) - np.min(sub_strategy[-1]))
        if epoch % PREDICTION_AVG_GAP == 0 and epoch != 0:
            prediction_trends_mean_avg.append(np.sum(prediction_trends_mean[epoch-PREDICTION_AVG_GAP:epoch])/ PREDICTION_AVG_GAP )
            prediction_trends_std_avg.append(np.sum(prediction_trends_std[epoch-PREDICTION_AVG_GAP:epoch]) / PREDICTION_AVG_GAP )
            prediction_trends_range_avg.append(np.sum(prediction_trends_range[epoch-PREDICTION_AVG_GAP:epoch]) / PREDICTION_AVG_GAP )

        if LOG_ON_MAIN:
            print('2K+1\'s sub strategy = ',sub_strategy)
        # ************************************************TSO-OPTIMAL**********************************************#
        obj_value_with_substrategy = []
        obj_value_with_substrategy_cvx = []
        t0_temp = []
        t0_temp_cvx = []
        #将上面预测出来的2K+1个决策逐一带入到TSO中，求出每个策略对应的时延值
        for m in sub_strategy:
            delay = TSO(h_origin, m, EPSILON, G, L)
            obj_value_with_substrategy.append(delay)
            # delay1, tt1 = cvx_all_local(h / TRICK_VALUE, m, epsilon, G, L,N)
            if CVX_ON:
                delay_cvx, tt_cvx = t_delay_of_cvx(h_origin, m, EPSILON, G, L, N)
                obj_value_with_substrategy_cvx.append(delay_cvx)
                t0_temp_cvx.append(tt_cvx)
        # ************************************************ENCODE**********************************************#
        #找出最小时延所对应的策略，然后和input编码在一起，放入到缓存中
        mem.encode(h_standard, sub_strategy[np.argmin(obj_value_with_substrategy)])
        if LOG_ON_MAIN:
            print('obj_value_with_substrategy:%s'%obj_value_with_substrategy)
            print('t0_temp:%s'%t0_temp)
            print('np.argmin(obj_value_with_substrategy):%s'%np.argmin(obj_value_with_substrategy))
            print('sub_strategy[np.argmin(obj_value_with_substrategy)]:%s'%sub_strategy[np.argmin(obj_value_with_substrategy)])
        obj_opt_arr.append(np.min(obj_value_with_substrategy))
        if CVX_ON:
            obj_opt_arr_cvx.append(np.min(obj_value_with_substrategy_cvx))
        # t0_opt_arr.append(t0_temp[np.argmin(obj_value_with_substrategy)])
        # ************************************************TEST**********************************************#
        #进行测试并记录测试数据
        if epoch % TRAINING_INTERVAL == 0:

            h_standard_for_test = CHANNEL_GAINS_STANDARDIZED[SPLIT_INDEX + 10, :]
            h_origin_for_test = CHANNEL_GAINS_ORIGIN[SPLIT_INDEX + 10, :]

            # TSO TEST
            DROO_TSO_start_time = time.time()  # 计算每次DROO-TSO的用时时长
            strategy = mem.decode_test(h_standard_for_test, K, N, DECODER_MODE)
            delay_test_TSO = TSO(h_origin_for_test, strategy, EPSILON, G, L)
            DROO_TSO_process_time_arr.append(time.time() - DROO_TSO_start_time)

            # ALL OFFLOAD TEST
            delay_test_alloff = t_delay_of_all_offload_with_stable_power(h_origin_for_test, L)

            # ALL LOCAL TEST
            delay_test_alllocal, *_ = t_delay_of_all_local_computing(h_origin_for_test, EPSILON, G, L)

            # BINARY OFFLOAD TEST
            BINARY_start_time = time.time()
            sub_binary_strategy = generate_binary_policy(N)
            min_delay = []
            for s in sub_binary_strategy:
                delay4 = t_delay_of_binary_offloading(h_origin_for_test, EPSILON, G, L, s)
                min_delay.append(delay4)
            delay_test_binary = np.min(min_delay)
            BINARY_process_time_arr.append(time.time() - BINARY_start_time)

            TSO_delay_arr.append(delay_test_TSO)
            binary_delay_arr.append(delay_test_binary)
            alloff_delay_arr.append(delay_test_alloff)
            alllocal_delay_arr.append(delay_test_alllocal)
            tso_strategy_history.append(strategy)
            binary_strategy_history.append(sub_binary_strategy[np.argmin(min_delay)])

    # ************************************************FOR END**********************************************#

    # ************************************************PLOT**********************************************#
    if CVX_ON:
        plot_data(np.arange(len(obj_opt_arr/obj_opt_arr_cvx)),obj_opt_arr/obj_opt_arr_cvx,
                  'Time Frames', 'ratio ',"ratio")

    # ************************************************SAVEDATA**********************************************#
    save_list_to_txt(TSO_delay_arr, PIC_PATH, "TSO_delay_arr")
    save_list_to_txt(binary_delay_arr, PIC_PATH, "binary_delay_arr")
    save_list_to_txt(alloff_delay_arr, PIC_PATH, "alloff_delay_arr")
    save_list_to_txt(alllocal_delay_arr, PIC_PATH, "alllocal_delay_arr")
    save_list_to_txt(DROO_TSO_process_time_arr, PIC_PATH, "DROO_TSO_process_time_arr")
    save_list_to_txt(BINARY_process_time_arr, PIC_PATH, "BINARY_process_time_arr")
    save_list_to_txt(mem.cost_arr, PIC_PATH, "cost_arr")
    save_list_to_txt(obj_opt_arr, PIC_PATH, "obj_opt_arr")
    # save_list_to_txt(t0_opt_arr, PIC_PATH, "t0_opt_arr")
    # save_list_to_txt(tloc, PIC_PATH, "tloc")
    # save_list_to_txt(toff, PIC_PATH, "toff")
    # save_list_to_txt(E_total, PIC_PATH, "E_total")
    # save_list_to_txt(E_off, PIC_PATH, "E_off")
    # save_list_to_txt(E_loc, PIC_PATH, "E_loc")
    save_list_to_txt(strategy, PIC_PATH, "strategy")
    save_list_to_txt(tso_strategy_history, PIC_PATH, "tso_strategy_history")
    save_list_to_txt(binary_strategy_history, PIC_PATH, "binary_strategy_history")
    save_list_to_txt(prediction_trends_mean, PIC_PATH, "prediction_trends_mean")
    save_list_to_txt(prediction_trends_mean_avg, PIC_PATH, "prediction_trends_mean_avg")
    save_list_to_txt(prediction_trends_std, PIC_PATH, "prediction_trends_std")
    save_list_to_txt(prediction_trends_std_avg, PIC_PATH, "prediction_trends_std_avg")
    save_list_to_txt(prediction_trends_range, PIC_PATH, "prediction_trends_range")
    save_list_to_txt(prediction_trends_range_avg, PIC_PATH, "prediction_trends_range_avg")
    # save_value_to_txt(tt, PIC_PATH, "tt")
    # save_value_to_txt(delay1, PIC_PATH, "delay1")
    save_value_to_txt(TRAINING_INTERVAL, PIC_PATH, "TRAINING_INTERVAL")
    save_value_to_txt(TOTAL_TIME, PIC_PATH, "TOTAL_TIME")
    # save_value_to_txt(sampling[0], PIC_PATH, "sampling0")
    # save_value_to_txt(sampling[1], PIC_PATH, "sampling1")
