# 此类的作用是将初始的策略分裂为多个邻近的策略

import numpy as np
from config import TOTAL_TIME, TRAINING_INTERVAL

# 修改后的噪声参数配置
SIGMA_INIT = 1  # 增大初始探索幅度
SIGMA_DECAY_FAST = 0.95  # 快速衰减速度 0.95
SIGMA_DECAY_SLOW = 0.99  # 缓慢衰减速度 0.99
MIN_SIGMA = 0.1  # 设置最小噪声
DECAY_BOUNDRY = TOTAL_TIME / 5 # 以DECAY_BOUNDRY为界限，前DECAY_BOUNDRY步快速衰减，DECAY_BOUNDRY后缓慢衰减
DECAY_INTERVAL = TOTAL_TIME/ TRAINING_INTERVAL / 50 # 每隔DECAY_INTERVAL步衰减一次
# BOUNDARY_THRESH = 0.000001  # 边界判断阈值

def get_current_sigma(epoch):
    # # 分段衰减策略
    # if epoch < DECAY_BOUNDRY:
    #     return max(SIGMA_INIT * (SIGMA_DECAY_FAST ** (epoch // DECAY_INTERVAL)), SIGMA_INIT/2)  # 前DECAY_BOUNDRY步快速衰减
    # else:
    #     return max(SIGMA_INIT/2 * (SIGMA_DECAY_SLOW ** ((epoch - DECAY_BOUNDRY) // DECAY_INTERVAL)), MIN_SIGMA)  # 后续缓慢衰减
    return 0.5

def generate_perturbations(
        y_base: np.ndarray,
        epoch: int,
        num_perturb: int = 20,
        adaptive_noise: bool = True
) -> np.ndarray:
    """
    生成邻近预测值

    参数：
    - y_base: 基础预测值数组，形状(10,)
    - num_perturb: 需要生成的扰动数量（默认20）
    - epoch: 用于指明当前运行到第几轮
    - adaptive_noise: 是否启用边界感知噪声（默认True）

    返回：
    - 形状为(num_perturb+1, 10)的数组，包含原始值和所有扰动值
    """

    # 生成噪声，指数衰减法
    # sigma = SIGMA_INIT * (SIGMA_DECAY_SLOW ** (epoch // DECAY_INTERVAL))

    # 生成噪声，分阶段衰减法
    sigma = get_current_sigma(epoch)

    # 创建结果容器
    perturbations = np.empty((num_perturb + 1, y_base.shape[0]))
    perturbations[0] = y_base.copy()

    # 批量生成噪声（向量化操作）
    noise = np.random.randn(num_perturb, y_base.shape[0]) * sigma
    # print("noise=",noise)

    # 允许所有方向扰动，并将结果限制在0～1
    perturbations[1:] = np.clip(y_base + noise, 0, 1)

    # 强制包含至少一个边界值样本（确保探索能力）
    # perturbations[-1] = np.where(np.random.rand(*y_base.shape) < 0.5, 1, 0)

    return perturbations

#废弃 duplicate
def get_policy_v1(base_policy, policy_num=20):
    mean = np.sum(base_policy)/len(base_policy)
    up_gap = 1-mean
    down_gap = mean
    polies = []
    if mean == 0:
        up_num = int(np.floor(up_gap * policy_num))
        up_step = up_gap / up_num
        for i in range(up_num):
            polies.append(base_policy + i * up_step)
        polies[:] = np.clip(polies, 0, 1)
        polies.append(base_policy)
    elif mean == 1:
        down_num = policy_num
        down_step = down_gap / down_num
        for j in range(down_num):
            polies.append(base_policy - j * down_step)
        polies[:] = np.clip(polies, 0, 1)
        polies.append(base_policy)
    else:
        up_num = int(np.floor(up_gap*policy_num))
        down_num = policy_num-up_num

        up_step = up_gap/up_num
        down_step = down_gap/down_num

        for i in range(up_num):
            polies.append(base_policy + i*up_step)

        for j in range(down_num):
            polies.append(base_policy - j*down_step)

        polies[:] = np.clip(polies, 0, 1)

        polies.append(base_policy)

    return polies

def get_policy_v2(base_policy, policy_num=20):
    mean = np.sum(base_policy)/len(base_policy)
    up_gap = 1-mean
    down_gap = mean
    polies = []
    up_num = int(policy_num / 2)
    down_num = int(policy_num / 2)

    up_step = up_gap/up_num
    down_step = down_gap/down_num

    for i in range(up_num+3):
        polies.append(base_policy + (i+1)*up_step)

    for j in range(down_num+3):
        polies.append(base_policy - (j+1)*down_step)

    polies[:] = np.clip(polies, 0, 1)

    polies.append(base_policy)

    return polies
