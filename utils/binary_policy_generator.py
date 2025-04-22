import itertools
import numpy as np

#此函数用于生成n个节点的所有二分卸载策略
def generate_binary_policy(n):
    # 生成所有可能的二进制策略
    all_policies = list(itertools.product([0, 1], repeat=n))

    # 将列表转换为NumPy数组，并将数据类型设置为float
    policies_array = np.array(all_policies, dtype=float)

    # # 将所有的0替换为0.000001
    # policies_array[policies_array == 0] = 0.001
    # policies_array[policies_array == 1] = 0.999

    return policies_array