from config import *
import numpy as np
import cvxpy as cp

def TSO(h, M, epsilon, G, L):

    # 设置 NumPy 的错误处理方式，将警告提升为异常
    np.seterr(divide='raise', invalid='raise')

    ##################################################################################################################
    # t0_opt 用于表示最优充电时间，标量
    # T_delay 用于表示系统整体时延，标量
    # f_opt 最优频率，数组
    # t_off 卸载时间，数组
    # t_loc 本地计算时间，数组

    # principle_1.M中可能会出现元素1，意味着不需要本地计算，则应把这些节点剔除后进行t_loc的计算
    # principle_2.另外由于本文采用了"数据丢弃0容忍"，所以每个节点的数据必须处理完，所以M中为1对应的节点的充电时间t0就成为充电时间下界t0_lower_bound，必须满足
    # principle_3.最后将M中非1节点的充电时间与1节点相比，取其中的最大值即得到最终应得的充电时间，然后继续优化
    # principle_4.整体时延 = 充电时间 + np.maximum(sum(t_off), max(t_loc))
    ##################################################################################################################

    contains_one = np.any(M == 1)  # 是否包含1
    all_ones = np.all(M == 1)  # 是否全为1
    all_zeros = np.all(M == 0)  # 是否全为0

    if all_ones:# 全为1
        # 变为全卸载，直接调用all_offload_with_stable_power()函数即可得到T_delay
        T_delay = t_delay_of_all_offload_with_stable_power(h, L)
    elif all_zeros:  # 全为0
        T_delay, *_ = t_delay_of_all_local_computing(h, epsilon, G, L)
    elif not contains_one:# 没有1,但不全为0
        # 直接调用t_delay_without_alloffload_node()函数即可得到T_delay
        T_delay = t_delay_without_alloffload_node(h, epsilon, G, L, M)
    else:# 部分为1

        # 先进行参数初始化计算
        t_off = M * L / BANDWIDTH / np.log2(1 + P_WD_MAX_W * h / XIGEMA_2_W)  # 为固定值，数组

        # 根据 principle_2 ，先计算出全卸载节点对应的充电时间，作为充电时间下界
        mask_is_1 = M == 1  # 创建一个布尔掩码，表示M中等于1的位置；接着使用布尔掩码来过滤值为1的数至新数组
        h_is_1 = h[mask_is_1]
        M_is_1 = M[mask_is_1]
        L_is_1 = L[mask_is_1]
        t0_lower_bound = np.max(M_is_1 * L_is_1 * P_WD_MAX_W / BANDWIDTH / np.log2(1 + P_WD_MAX_W * h_is_1 / XIGEMA_2_W) / RECHARGE_EFFICIENCY / AP_POWER_W / h_is_1)  # 为固定值，标量scalar

        # 剔除M中为1的元素，并同步剔除h, epsilon, G, L中对应位置的元素，以用来进行本地计算时延的优化
        mask_not_1 = M != 1  # 创建一个布尔掩码，表示M中不等于1的位置；接着使用布尔掩码来过滤数组
        h = h[mask_not_1]
        M = M[mask_not_1]
        epsilon = epsilon[mask_not_1]
        G = G[mask_not_1]
        L = L[mask_not_1]
        E_off = M * L * P_WD_MAX_W / BANDWIDTH / np.log2(1 + P_WD_MAX_W * h / XIGEMA_2_W)  # 为固定值

        # (0)
        a_i = E_off
        b_i = np.sqrt(epsilon * np.power(1 - M, 3) * np.power(L, 3) * np.power(G, 3))
        c_i = RECHARGE_EFFICIENCY * AP_POWER_W * h
        # (1)
        t0_opt = np.max(E_off / RECHARGE_EFFICIENCY / AP_POWER_W / h) + 1e3  # 得到初始的系统充电时间
        # (2)
        # 由于F_WD_MAX，所以当t0_opt足够大时，t_loc会达到最小值，但不会随着t0_opt的增大而无限减小，所以在此处就引入了计算频率f的限制条件
        # 所以不能将t_loc简单的表示为： t_loc = b_i / np.sqrt(c_i * t0_opt - a_i)，而应该结合F_WD_MAX展开分析
        t_loc = []
        t0_loc = (epsilon * (1 - M) * L * G * F_WD_MAX * F_WD_MAX + E_off) / (RECHARGE_EFFICIENCY * AP_POWER_W * h)
        for i in range(len(M)):
            if t0_opt >= t0_loc[i]:
                t_loc.append((1 - M[i]) * L[i] * G[i] / F_WD_MAX / F_WD_MAX)
            else:
                t_loc.append(b_i[i] / np.sqrt(c_i[i] * t0_opt - a_i[i]))

        # 根据 t_loc 得到 t_loc_worst 的值，再结合 t_off 和 t0_opt 计算出 T_delay
        t_loc_worst = np.max(t_loc)
        T_delay = t0_opt + np.maximum(t_loc_worst, np.sum(t_off))

        # (3)
        w_i = np.argmax(t_loc)  # 确定出哪个节点是worst节点，得到索引值w_i
        # (7) 得到除了w以外节点中a_i/c_i最大值作为m
        temp1 = a_i / c_i
        awcw = temp1[w_i]  # w节点的a_w/c_w值
        m_i = np.argmax(temp1)  # a_i/c_i最大值对应的索引值
        amcm = temp1[m_i]  # m节点的a_m/c_m值
        # (4~6) 接着，以w_i为基础进行t0的优化，得到f(t0)的最小值点t0_min和函数曲线变化临界点t0_upper
        # t0_*
        t0_min = (a_i[w_i] + np.power(2 / b_i[w_i] / c_i[w_i], 1.5)) / c_i[w_i]
        # t0_'
        t0_upper = (epsilon[w_i] * (1 - M[w_i]) * L[w_i] * G[w_i] * np.power(F_WD_MAX, 2) + a_i[w_i]) / c_i[w_i]
        # t0_^
        t0_hat = (t0_upper if t0_min >= t0_upper else t0_min)

        if LOG_ON_OPTIMIZATION:
            print('******************OPTIMIZATION PARA******************')
            print("M without 1:",M)
            print("t_off:", t_off)
            print("t0_lower_bound:", t0_lower_bound)
            print("h:", h)
            print("epsilon:", epsilon)
            print("G:", G)
            print("L:", L)
            print("E_off:", E_off)
            print("t_loc:", t_loc)
            print('t0_opt initial value:%s' % t0_opt)
            print('a_i:%s' % a_i)
            print('b_i:%s' % b_i)
            print('c_i:%s' % c_i)
            print("t_loc_worst:", t_loc_worst)
            print("T_delay:", T_delay)
            print('w_i:%s'%w_i)
            print('temp1:%s'%temp1)
            print('awcw:%s'%awcw)
            print('m_i:%s'%m_i)
            print('amcm:%s'%amcm)
            print('t0_min:%s'%t0_min)
            print('t0_upper:%s'%t0_upper)
            print('t0_hat:%s'%t0_hat)
            print('******************OPTIMIZATION PARA******************')

        # 需要将 t0_lower_bound 与 amcm, awcw, t_hat 比大小，先大致分为三种情况：
        if t0_lower_bound < min(amcm, awcw, t0_min, t0_upper):
            # 则计算所得的就是最终的值
            T_delay = t_delay_without_alloffload_node(h,epsilon,G,L,M)
        elif t0_lower_bound > max(amcm, awcw, t0_min, t0_upper):
            # 则 t0_opt = t0_lower_bound
            # 将 t0_opt 带入 f_(w) 求出最终的 T_delay
            t0_opt = t0_lower_bound
            t_loc_worst = (1 - M[w_i]) * L[w_i] * G[w_i] / F_WD_MAX
            if t_loc_worst < np.sum(t_off):
                T_delay = t0_opt + np.sum(t_off)
            else:
                T_delay = t0_opt + t_loc_worst
        else:
            # 这个情况比较复杂需要结合下方的(8)(9)(10)分类讨论
            #--------------------------------------------------TSO优化算法开始--------------------------------------------------#
            # (8)
            if amcm<=awcw and awcw<t0_hat:

                if t0_min < t0_upper:
                    if t0_lower_bound <= t0_min:
                        t0_opt = t0_hat
                        t_loc_worst = b_i[w_i] / np.sqrt(c_i[w_i] * t0_opt - a_i[w_i])
                        # (8.1)
                        if t_loc_worst < np.sum(t_off):  # 有继续优化的空间，开启第二阶段优化
                            t0_opt = E_off[w_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] + epsilon[w_i] * np.power(1 - M[w_i], 3) * np.power(L[w_i], 3) * np.power(G[w_i],3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] / np.power(np.sum(t_off), 2)
                            # 如果优化后的 t0_opt 小于 t0_lower_bound，则 t0_opt = t0_lower_bound
                            if t0_opt <= t0_lower_bound:
                                t0_opt = t0_lower_bound
                            T_delay = t0_opt + np.sum(t_off)
                        else:  # 没有继续优化的空间，不需开启第二阶段优化
                            # 更新T_delay的值
                            T_delay = t0_opt + t_loc_worst
                    else: # t0_min < t0_lower_bound <= t0_upper
                        t0_opt = t0_lower_bound
                        t_loc_worst = b_i[w_i] / np.sqrt(c_i[w_i] * t0_opt - a_i[w_i])
                        if t_loc_worst < np.sum(t_off):
                            T_delay = t0_opt + np.sum(t_off)
                        else:
                            T_delay = t0_opt + t_loc_worst
                else:
                    if t0_lower_bound <= t0_upper:
                        t0_opt = t0_hat
                        t_loc_worst = b_i[w_i] / np.sqrt(c_i[w_i] * t0_opt - a_i[w_i])
                        # (8.1)
                        if t_loc_worst < np.sum(t_off):  # 有继续优化的空间，开启第二阶段优化
                            t0_opt = E_off[w_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] + epsilon[w_i] * np.power(1 - M[w_i], 3) * np.power(L[w_i], 3) * np.power(G[w_i],3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] / np.power(np.sum(t_off), 2)
                            # 如果优化后的 t0_opt 小于 t0_lower_bound，则 t0_opt = t0_lower_bound
                            if t0_opt <= t0_lower_bound:
                                t0_opt = t0_lower_bound
                            T_delay = t0_opt + np.sum(t_off)

                        else:  # 没有继续优化的空间，不需开启第二阶段优化
                            # 更新T_delay的值
                            T_delay = t0_opt + t_loc_worst
                    else: # t0_upper < t0_lower_bound <= t0_min
                        t0_opt = t0_lower_bound
                        t_loc_worst = (1-M[w_i]) * L[w_i] * G[w_i] / F_WD_MAX
                        if t_loc_worst < np.sum(t_off):
                            T_delay = t0_opt + np.sum(t_off)
                        else:
                            T_delay = t0_opt + t_loc_worst

            # (9)
            elif awcw<t0_hat and t0_hat<=amcm:

                if t0_upper <= t0_min:
                    t0_bar = amcm + np.power(b_i[m_i],2)*F_WD_MAX**2/c_i[m_i]/np.power(1-M[w_i],2)/np.power(L[w_i],2)/np.power(G[w_i],2)
                    t0_opt = t0_bar
                    t_loc_worst = (1-M[w_i]) * L[w_i] * G[w_i] / F_WD_MAX
                else:
                    if awcw<t0_min and t0_min<t0_upper and t0_upper<amcm:
                        t0_bar = amcm + np.power(b_i[m_i],2)*F_WD_MAX**2/c_i[m_i]/np.power(1-M[w_i],2)/np.power(L[w_i],2)/np.power(G[w_i],2)
                        t0_opt = t0_bar
                        t_loc_worst = (1-M[w_i]) * L[w_i] * G[w_i] / F_WD_MAX
                    elif awcw<t0_min and t0_min<amcm and amcm<t0_upper:
                        t0_bar = (b_i[w_i]**2*a_i[m_i]-b_i[m_i]**2*a_i[w_i])/(b_i[w_i]**2*c_i[m_i]-b_i[m_i]**2*c_i[w_i])
                        t0_opt = t0_bar
                        t_loc_worst = np.sqrt(epsilon[w_i] * np.power((1-M[w_i]),3) * np.power(L[w_i],3) * np.power(G[w_i],3) / (RECHARGE_EFFICIENCY * AP_POWER_W * h[w_i] * t0_opt - E_off[w_i]))
                    else:
                        print('理论分析不完善')
                        assert(True)
                # (9.1)
                # 不需要计算全部的t_loc值，只需要计算t_loc_w的值即可（因为最终是为了得到T_delay）, 但要注意函数表达式的变化
                # 由于 t0_opt 非常接近于 a_m / c_m , 导致 b_i / np.sqrt(c_i * t0_opt - a_i) 的分母较小，
                # 系统精度不够会将其认为是0，导致出现除数为0的问题。 所以不直接使用 t_loc = b_i / np.sqrt(c_i * t0_opt - a_i) 求解，
                # 而是根据 t0_opt 是由 f(w) 与 f(m) 的交点得出的，从而  f(w)=t0_opt+t_loc_w == f(m)=t0_opt+t_loc_m,
                # 所以求出t0_opt处的t_loc_w即为t_loc_worst

                if t_loc_worst < np.sum(t_off):#有继续优化的空间，开启第二阶段优化
                    t0_opt = E_off[m_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] + epsilon[m_i] * np.power(1 - M[m_i], 3) * np.power(L[m_i],3) * np.power(G[m_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] / np.power(np.sum(t_off), 2)
                    # 更新T_delay的值
                    T_delay = t0_opt + np.sum(t_off)
                else:#没有继续优化的空间，不需开启第二阶段优化
                    # 更新T_delay的值
                    T_delay = t0_opt + t_loc_worst

            # (10)
            elif awcw<amcm and amcm<t0_hat:

                if t0_lower_bound <= amcm:
                    t0_opt = t0_hat
                    t_loc_worst = b_i[w_i] / np.sqrt(c_i[w_i] * t0_opt - a_i[w_i])
                    if t_loc_worst < np.sum(t_off):  # 有继续优化的空间，开启第二阶段优化
                        t0_bar = (b_i[w_i] ** 2 * a_i[m_i] - b_i[m_i] ** 2 * a_i[w_i]) / (b_i[w_i] ** 2 * c_i[m_i] - b_i[m_i] ** 2 * c_i[w_i])
                        f_t0 = t0_bar + b_i[w_i] / np.sqrt(c_i[w_i] * t0_bar - a_i[w_i])
                        g_t0 = t0_bar + np.sum(t_off)
                        # (10.1)
                        if f_t0 <= g_t0:
                            t0_opt = E_off[m_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] + epsilon[m_i] * np.power(1 - M[m_i], 3) * np.power(L[m_i], 3) * np.power(G[m_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] / np.power(np.sum(t_off), 2)
                        else:
                            t0_opt = E_off[w_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] + epsilon[w_i] * np.power(1 - M[w_i], 3) * np.power(L[w_i], 3) * np.power(G[w_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] / np.power(np.sum(t_off), 2)
                        # 更新T_delay的值
                        T_delay = t0_opt + np.sum(t_off)
                    else:  # 没有继续优化的空间，不需开启第二阶段优化
                        # 更新T_delay的值
                        T_delay = t0_opt + t_loc_worst
                else:
                    if t0_min < t0_upper:
                        if t0_lower_bound <= t0_min:
                            t0_opt = t0_hat
                            t0_bar = (b_i[w_i] ** 2 * a_i[m_i] - b_i[m_i] ** 2 * a_i[w_i]) / (b_i[w_i] ** 2 * c_i[m_i] - b_i[m_i] ** 2 * c_i[w_i])
                            t_loc_worst = b_i[w_i] / np.sqrt(c_i[w_i] * t0_opt - a_i[w_i])

                            if t_loc_worst < np.sum(t_off):  # 有继续优化的空间，开启第二阶段优化
                                f_t0 = t0_bar + b_i[w_i] / np.sqrt(c_i[w_i] * t0_bar - a_i[w_i])
                                g_t0 = t0_bar + np.sum(t_off)
                                # (10.1)
                                if f_t0 <= g_t0:
                                    t0_opt = E_off[m_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] + epsilon[m_i] * np.power(1 - M[m_i], 3) * np.power(L[m_i], 3) * np.power(G[m_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] / np.power(np.sum(t_off), 2)
                                else:
                                    t0_opt = E_off[w_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] + epsilon[w_i] * np.power(1 - M[w_i], 3) * np.power(L[w_i], 3) * np.power(G[w_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] / np.power(np.sum(t_off), 2)
                                if t0_lower_bound > t0_opt:
                                    t0_opt = t0_lower_bound
                                T_delay = t0_opt + np.sum(t_off)
                            else:  # 没有继续优化的空间，不需开启第二阶段优化
                                # 更新T_delay的值
                                T_delay = t0_opt + t_loc_worst
                        else:
                            t0_opt = t0_lower_bound
                            t_loc_worst = b_i[w_i] / np.sqrt(c_i[w_i] * t0_opt - a_i[w_i])
                            if t_loc_worst < np.sum(t_off):
                                T_delay = t0_opt + np.sum(t_off)
                            else:
                                T_delay = t0_opt + t_loc_worst
                    else:
                        if t0_lower_bound <= t0_upper:
                            t0_opt = t0_hat
                            t0_bar = (b_i[w_i] ** 2 * a_i[m_i] - b_i[m_i] ** 2 * a_i[w_i]) / (b_i[w_i] ** 2 * c_i[m_i] - b_i[m_i] ** 2 * c_i[w_i])
                            t_loc_worst = b_i[w_i] / np.sqrt(c_i[w_i] * t0_opt - a_i[w_i])

                            if t_loc_worst < np.sum(t_off):  # 有继续优化的空间，开启第二阶段优化
                                f_t0 = t0_bar + b_i[w_i] / np.sqrt(c_i[w_i] * t0_bar - a_i[w_i])
                                g_t0 = t0_bar + np.sum(t_off)
                                # (10.1)
                                if f_t0 <= g_t0:
                                    t0_opt = E_off[m_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] + epsilon[m_i] * np.power(1 - M[m_i], 3) * np.power(L[m_i], 3) * np.power(G[m_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] / np.power(np.sum(t_off), 2)
                                else:
                                    t0_opt = E_off[w_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] + epsilon[w_i] * np.power(1 - M[w_i], 3) * np.power(L[w_i], 3) * np.power(G[w_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] / np.power(np.sum(t_off), 2)
                                if t0_lower_bound > t0_opt:
                                    t0_opt = t0_lower_bound
                                T_delay = t0_opt + np.sum(t_off)
                            else:  # 没有继续优化的空间，不需开启第二阶段优化
                                # 更新T_delay的值
                                T_delay = t0_opt + t_loc_worst
                        else:
                            t0_opt = t0_lower_bound
                            t_loc_worst = (1 - M[w_i]) * L[w_i] * G[w_i] / F_WD_MAX
                            if t_loc_worst < np.sum(t_off):
                                T_delay = t0_opt + np.sum(t_off)
                            else:
                                T_delay = t0_opt + t_loc_worst

            else:
                print('something is wrong...')
                print('amcm:%s'%amcm)
                print('awcw:%s'%awcw)
                print('to_hat:%s'%t0_hat)
                assert (True)
            # --------------------------------------------------TSO优化算法结束--------------------------------------------------#

    return T_delay

# 这个函数是用于计算"所有节点均为全卸载"的最优T_delay
def t_delay_of_all_offload_with_stable_power(h, L):
    t_off = L / BANDWIDTH / np.log2(1 + P_WD_MAX_W * h / XIGEMA_2_W)  # 为固定值，数组

    # 计算出全卸载节点对应的充电时间，作为充电时间下界
    t0_lower_bound = np.max(L * P_WD_MAX_W / BANDWIDTH / np.log2(1 + P_WD_MAX_W * h / XIGEMA_2_W) / RECHARGE_EFFICIENCY / AP_POWER_W / h)  # 为固定值，标量scalar

    T_delay = t0_lower_bound + np.sum(t_off)

    return T_delay

# 这个函数是用于计算不包含"全卸载节点"的最优T_delay， 但需注意⚠️的是不包含全为0的情况
def t_delay_without_alloffload_node(h, epsilon, G, L, M):

    t_off = M * L / BANDWIDTH / np.log2(1 + P_WD_MAX_W * h / XIGEMA_2_W)  # 为固定值，数组
    E_off = M * L * P_WD_MAX_W / BANDWIDTH / np.log2(1 + P_WD_MAX_W * h / XIGEMA_2_W)  # 为固定值

    # (0)
    a_i = E_off
    b_i = np.sqrt(epsilon * np.power(1 - M, 3) * np.power(L, 3) * np.power(G, 3))
    c_i = RECHARGE_EFFICIENCY * AP_POWER_W * h

    # 初始化 t0
    t0_opt = 1e5  # 得到初始的系统充电时间(需足够大)

    # (2)
    # 由于F_WD_MAX，所以当t0_opt足够大时，t_loc会达到最小值，但不会随着t0_opt的增大而无限减小，所以在此处就引入了计算频率f的限制条件
    # 所以不能将t_loc简单的表示为： t_loc = b_i / np.sqrt(c_i * t0_opt - a_i)，而应该结合F_WD_MAX展开分析
    t_loc = []
    t0_loc = (epsilon * (1-M) * L * G * F_WD_MAX * F_WD_MAX + E_off) / (RECHARGE_EFFICIENCY * AP_POWER_W * h)
    for i in range(len(M)):
        if t0_opt >= t0_loc[i]:
            t_loc.append((1-M[i])*L[i]*G[i]/F_WD_MAX/F_WD_MAX)
        else:
            t_loc.append(b_i[i] / np.sqrt(c_i[i] * t0_opt - a_i[i]))

    # (3)
    w_i = np.argmax(t_loc)  # 确定出哪个节点是worst节点，得到索引值w_i
    # (7) 得到除了w以外节点中a_i/c_i最大值作为m
    temp1 = a_i / c_i
    awcw = temp1[w_i]  # w节点的a_w/c_w值
    m_i = np.argmax(temp1)  # a_i/c_i最大值对应的索引值
    amcm = temp1[m_i]  # m节点的a_m/c_m值
    # (4~6) 接着，以w_i为基础进行t0的优化，得到f(t0)的最小值点t0_min和函数曲线变化临界点t0_upper
    # t0_*
    t0_min = (a_i[w_i] + np.power(2 / b_i[w_i] / c_i[w_i], 1.5)) / c_i[w_i]
    # t0_'
    t0_upper = (epsilon[w_i] * (1 - M[w_i]) * L[w_i] * G[w_i] * np.power(F_WD_MAX, 2) + a_i[w_i]) / c_i[w_i]
    # t0_^
    t0_hat = (t0_upper if t0_min >= t0_upper else t0_min)

    # 根据 t_loc 得到 t_loc_worst 的值，再结合 t_off 和 t0_opt 计算出 T_delay
    t_loc_worst = np.max(t_loc)
    T_delay = t0_opt + t_loc_worst

    # --------------------------------------------------TSO优化算法开始--------------------------------------------------#
    # (8)
    if amcm <= awcw and awcw < t0_hat:
        t0_opt = t0_hat
        t_loc_worst = b_i[w_i] / np.sqrt(c_i[w_i] * t0_opt - a_i[w_i]) # 由于此时 t0_opt 一定小于等于 t0_upper，所以一定满足 F_WD_MAX约束
        # (8.1)
        if t_loc_worst < np.sum(t_off):  # 有继续优化的空间，开启第二阶段优化
            t0_opt = E_off[w_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] + epsilon[w_i] * np.power(1 - M[w_i],3) * np.power(L[w_i], 3) * np.power(G[w_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] / np.power(np.sum(t_off),2)
            # 更新T_delay的值
            T_delay = t0_opt + np.sum(t_off)
        else:  # 没有继续优化的空间，不需开启第二阶段优化
            # 更新T_delay的值
            T_delay = t0_opt + t_loc_worst
    # (9)
    elif awcw < t0_hat and t0_hat <= amcm:
        if t0_upper <= t0_min:
            t0_bar = amcm + np.power(b_i[m_i], 2) * F_WD_MAX ** 2 / c_i[m_i] / np.power(1 - M[w_i], 2) / np.power(L[w_i], 2) / np.power(G[w_i], 2)
            t0_opt = t0_bar
            t_loc_worst = (1 - M[w_i]) * L[w_i] * G[w_i] / F_WD_MAX
        else:
            if awcw < t0_min and t0_min < t0_upper and t0_upper < amcm:
                t0_bar = amcm + np.power(b_i[m_i], 2) * F_WD_MAX ** 2 / c_i[m_i] / np.power(1 - M[w_i], 2) / np.power(L[w_i], 2) / np.power(G[w_i], 2)
                t0_opt = t0_bar
                t_loc_worst = (1 - M[w_i]) * L[w_i] * G[w_i] / F_WD_MAX
            elif awcw < t0_min and t0_min < amcm and amcm < t0_upper:
                t0_bar = (b_i[w_i] ** 2 * a_i[m_i] - b_i[m_i] ** 2 * a_i[w_i]) / (b_i[w_i] ** 2 * c_i[m_i] - b_i[m_i] ** 2 * c_i[w_i])
                t0_opt = t0_bar
                t_loc_worst = np.sqrt(epsilon[w_i] * np.power((1 - M[w_i]), 3) * np.power(L[w_i], 3) * np.power(G[w_i], 3) / (RECHARGE_EFFICIENCY * AP_POWER_W * h[w_i] * t0_opt - E_off[w_i]))
            else:
                print('理论分析不完善')
                assert (True)
        # (9.1)
        # 不需要计算全部的t_loc值，只需要计算t_loc_w的值即可（因为最终是为了得到T_delay）, 但要注意函数表达式的变化
        # 由于 t0_opt 非常接近于 a_m / c_m , 导致 b_i / np.sqrt(c_i * t0_opt - a_i) 的分母较小，
        # 系统精度不够会将其认为是0，导致出现除数为0的问题。 所以不直接使用 t_loc = b_i / np.sqrt(c_i * t0_opt - a_i) 求解，
        # 而是根据 t0_opt 是由 f(w) 与 f(m) 的交点得出的，从而  f(w)=t0_opt+t_loc_w == f(m)=t0_opt+t_loc_m,
        # 所以求出t0_opt处的t_loc_w即为t_loc_worst

        if t_loc_worst < np.sum(t_off):  # 有继续优化的空间，开启第二阶段优化
            t0_opt = E_off[m_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] + epsilon[m_i] * np.power(1 - M[m_i],3) * np.power(L[m_i], 3) * np.power(G[m_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] / np.power(np.sum(t_off),2)
            # 更新T_delay的值
            T_delay = t0_opt + np.sum(t_off)
        else:  # 没有继续优化的空间，不需开启第二阶段优化
            # 更新T_delay的值
            T_delay = t0_opt + t_loc_worst

    # (10)
    elif awcw < amcm and amcm < t0_hat:
        t0_opt = t0_hat
        t_loc_worst = b_i[w_i] / np.sqrt(c_i[w_i] * t0_opt - a_i[w_i])
        if t_loc_worst < np.sum(t_off):  # 有继续优化的空间，开启第二阶段优化
            t0_bar = (b_i[w_i] ** 2 * a_i[m_i] - b_i[m_i] ** 2 * a_i[w_i]) / (b_i[w_i] ** 2 * c_i[m_i] - b_i[m_i] ** 2 * c_i[w_i])
            f_t0 = t0_bar + b_i[w_i] / np.sqrt(c_i[w_i] * t0_bar - a_i[w_i])
            g_t0 = t0_bar + np.sum(t_off)
            # (10.1)
            if f_t0 <= g_t0:
                t0_opt = E_off[m_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] + epsilon[m_i] * np.power(1 - M[m_i], 3) * np.power(L[m_i], 3) * np.power(G[m_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[m_i] / np.power(np.sum(t_off), 2)
            else:
                t0_opt = E_off[w_i] / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] + epsilon[w_i] * np.power(1 - M[w_i], 3) * np.power(L[w_i], 3) * np.power(G[w_i], 3) / RECHARGE_EFFICIENCY / AP_POWER_W / h[w_i] / np.power(np.sum(t_off), 2)
            # 更新T_delay的值
            T_delay = t0_opt + np.sum(t_off)
        else:  # 没有继续优化的空间，不需开启第二阶段优化
            # 更新T_delay的值
            T_delay = t0_opt + t_loc_worst
    else:
        print('something is wrong...')
        print('amcm:%s' % amcm)
        print('awcw:%s' % awcw)
        print('to_hat:%s' % t0_hat)
        assert (True)
    # --------------------------------------------------TSO优化算法结束--------------------------------------------------#
    return T_delay

# 这个函数是用于计算"所有节点均为本地计算"的最优T_delay
def t_delay_of_all_local_computing(h, epsilon, G, L):

    # T_delay的表达式如下，求二阶导数后分析可知T_delay是关于f的凸函数
    # T_delay = epsilon*L*G*f*f/(RECHARGE_EFFICIENCY * AP_POWER_W * h) + L*G/f

    # 求出T_delay的极值点（f_star，T_delay_min_theoretical），但是f_star不一定能取到，需要和F_WD_MAX比对进行分类讨论
    f_star = ((RECHARGE_EFFICIENCY * AP_POWER_W * h) / (2 * epsilon)) ** (1 / 3)
    T_delay_min_theoretical = 3 * ( (epsilon * L**3 * G**3) / (4 * RECHARGE_EFFICIENCY * AP_POWER_W * h) ) ** (1/3)

    T_delay_min = []
    for i in range(len(f_star)):
        if f_star[i]<=F_WD_MAX:
            #可以取到极值点
            T_delay_min.append(T_delay_min_theoretical[i])
        else:
            f_star[i] = F_WD_MAX
            T_delay_min.append(epsilon*L*G*F_WD_MAX*F_WD_MAX/(RECHARGE_EFFICIENCY * AP_POWER_W * h) + L*G/F_WD_MAX)

    t_loc = L*G/f_star
    t0 = epsilon*L*G*f_star*f_star/(RECHARGE_EFFICIENCY * AP_POWER_W * h)
    return np.max(T_delay_min), t_loc, t0

# 这个函数穷举了所有的二分卸载决策，并从中找到最优T_delay
def t_delay_of_binary_offloading(h, epsilon, G, L, binary_strategy):

    if np.all(binary_strategy == 1):
        T_delay = t_delay_of_all_offload_with_stable_power(h, L)
    elif np.all(binary_strategy == 0):
        T_delay, *_ = t_delay_of_all_local_computing(h, epsilon, G, L)
    else:
        mask_is_1 = binary_strategy == 1  # 创建一个布尔掩码，表示M中等于1的位置；接着使用布尔掩码来过滤值为1的数至新数组
        h_is_1 = h[mask_is_1]
        L_is_1 = L[mask_is_1]
        t0_lower_bound_offload = np.max(L_is_1 * P_WD_MAX_W / BANDWIDTH / np.log2(1 + P_WD_MAX_W * h_is_1 / XIGEMA_2_W) / RECHARGE_EFFICIENCY / AP_POWER_W / h_is_1)
        t_off = L_is_1 / BANDWIDTH / np.log2(1 + P_WD_MAX_W * h_is_1 / XIGEMA_2_W)

        mask_not_1 = binary_strategy == 0  # 创建一个布尔掩码，表示M中不等于1的位置；接着使用布尔掩码来过滤数组
        h_not_1 = h[mask_not_1]
        L_not_1 = L[mask_not_1]
        G_not_1 = G[mask_not_1]
        epsilon_not_1 = epsilon[mask_not_1]

        T_delay, t_loc, t0 = t_delay_of_all_local_computing(h_not_1, epsilon_not_1, G_not_1, L_not_1)

        t0_lower_bound_local = np.max(t0)
        t_local_worst = np.max(t_loc)

        T_delay = np.maximum(t0_lower_bound_local, t0_lower_bound_offload) + np.maximum(np.sum(t_off), t_local_worst)

    return T_delay

# 这里的计算方式也有问题，并不一定是要全功率的，这是一种偷懒的方法，将被舍弃
def t_delay_of_all_local_all_frequence(h, epsilon, G, L):
    T_delay = np.max(F_WD_MAX**2 * L*G*epsilon / RECHARGE_EFFICIENCY / AP_POWER_W / h ) + np.max(L*G/F_WD_MAX)
    return T_delay

def t_delay_of_cvx(h, M, epsilon, G, L, N,weights=[]):

    f_opt = cp.Variable(N)
    t0_opt = cp.Variable()
    tao = cp.Variable()

    T_delay =  0.0
    zero = np.zeros(N)
    z1 = P_WD_MAX_W * M * L / BANDWIDTH /np.log2(1+P_WD_MAX_W*h/XIGEMA_2_W)
    z2 = epsilon*(1-M)*L*G
    z3 = RECHARGE_EFFICIENCY*AP_POWER_W*h
    z4 = (1-M)*L*G
    z5 = np.sum(M * L / BANDWIDTH /np.log2(1+P_WD_MAX_W*h/XIGEMA_2_W))

    print('z1:%s'%z1)
    print('z2:%s'%z2)
    print('z3:%s'%z3)
    print('z4:%s'%z4)
    print('z5:%s'%z5)

    constraints = [t0_opt >= 0,
                   tao>=0,
                   f_opt >= 0,
                   f_opt <= F_WD_MAX,
                   z1 + cp.multiply(z2, cp.square(f_opt)) - cp.multiply(z3, t0_opt) <= 0,
                   t0_opt + cp.multiply(z4,cp.inv_pos(f_opt)) - tao <= 0,
                   t0_opt + z5 - tao <= 0
                   ]

    # Form objective.
    obj = cp.Minimize(tao)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)

    # Returns the optimal value.
    # ans = prob.solve(solver='CPLEX')
    ans = prob.solve(solver='CVXOPT')

    T_delay = tao

    if LOG_ON_OPTIMIZATION:
        print('******************start******************')
        print('tao is :%s'%tao)
        print('t0_opt is :%s'%t0_opt)
        print('******************end******************')

    return T_delay, t0_opt

def t_delay_of_cvx_all_local(h, M, epsilon, G, L, N,weights=[]):

    f_opt = cp.Variable(N)
    t0_opt = cp.Variable()
    tao = cp.Variable()

    T_delay =  0.0
    z3 = RECHARGE_EFFICIENCY*AP_POWER_W*h

    constraints = [-t0_opt <= 0,
                   -f_opt <= 0,
                   f_opt - F_WD_MAX <= 0,
                   cp.multiply(epsilon*L*G*10**19, cp.square(f_opt)) - cp.multiply(z3*10**19, t0_opt) <= 0,
                   t0_opt + cp.multiply(L*G,cp.inv_pos(f_opt)) - tao <= 0
                   ]

    # Form objective.
    obj = cp.Minimize(tao)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)

    # Returns the optimal value.
    # ans = prob.solve(solver='CPLEX')
    # ans = prob.solve(solver='CVXOPT')
    # ans = prob.solve(solver=cp.GLPK)
    ans = prob.solve(solver=cp.SCS)
    # ans = prob.solve(solver=cp.ECOS)
    # ans = prob.solve(solver=cp.OSQP)
    # ans = prob.solve(solver=cp.CBC)
    print("status:", prob.status)
    T_delay = tao.value

    if LOG_ON_OPTIMIZATION:
        print('******************start******************')
        print('tao is :%s'%tao.value)
        print('t0_opt is :%s'%t0_opt.value)
        print('******************end******************')

    return T_delay, t0_opt.value

