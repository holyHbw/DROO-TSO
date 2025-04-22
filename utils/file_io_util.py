import inspect
import re
import os
import pandas as pd

# 示例调用函数
# 示例用法
# array_example = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]
# save_array_to_csv(array=array_example, file_name='example.csv',
#                   column_names=['A', 'B', 'C'], save_path='./output')
def save_array_to_csv(array, file_name, column_names=None, save_path='.'):
    """
    将二维数组保存为CSV文件。
    参数:
    - array: 要保存的二维数组或列表的列表。
    - file_name: CSV文件的名称（包括扩展名.csv）。
    - column_names: 一维列表，包含每个列的名称。默认值为None，表示不添加列名。
    - save_path: 文件保存的路径。默认值为'.'，即当前目录。
    返回:
    - None
    """
    # 创建DataFrame
    df = pd.DataFrame(array, columns=column_names)

    # 如果save_path不是绝对路径，则转换为相对于当前工作目录的路径
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)

    # 确保目标目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 完整的文件路径
    full_file_path = os.path.join(save_path, file_name)

    # 根据是否提供了column_names决定是否添加header
    if column_names is None:
        df.to_csv(full_file_path, index=False, header=False)  # 不添加列名
    else:
        df.to_csv(full_file_path, index=False)  # 添加列名

    print(f"文件已保存至: {full_file_path}")

#假设读取出的数组都是不应该包含列名的（因为处理数据时有列名反而碍事，列名只是为了方便excel中查看）
#1.如果csv数组包含列名，则需删掉第一行keep_first_row=False
#2.如果csv数组不包含列名，则设置keep_first_row=True
def load_array_from_csv(file_path, keep_first_row=False):
    # 使用pandas读取csv文件
    if keep_first_row:
        df = pd.read_csv(file_path, header=None)
        # 因为没有列名，所以直接使用所有数据
        array_loaded = df.values.tolist()
    else:
        df = pd.read_csv(file_path)
        # 因为有列名，所以跳过第一行的实际数据
        array_loaded = df.values[0:].tolist()
    return array_loaded

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)',line)
        if m:
            return m.group(1)

def save_list_to_txt(list, store_path, file_name):
    mkdir(store_path)
    # 将列表保存到txt文件
    with open(os.path.join(store_path, file_name + ".txt"), 'w') as f:
        for rate in list:
            f.write("%s\n" % rate)

def save_value_to_txt(value, store_path, file_name):
    mkdir(store_path)
    with open(os.path.join(store_path, file_name + ".txt"), 'w') as f:
        f.write("%s \n" % value)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def openfile_getdata(path):
    data = []
    with open(path) as f:
        for line in f:
            data.extend([float(i) for i in line.split()])
    return data