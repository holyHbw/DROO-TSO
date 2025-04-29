from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn
from utils.plot_util import plot_data
import numpy as np
from utils.device_manager import DM
from utils.sub_policy_generator import generate_perturbations, get_policy_v2
from config import TOTAL_TIME, BATCH_SIZE, TRAINING_INTERVAL

print(torch.__version__)

# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 0

        # store training cost
        self.cost_arr = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

        # 新增代码 start -----------------------------------#
        # 新增学习率调度相关参数
        # self.warmup_steps = 100  # 预热步数
        # self.total_steps = TOTAL_TIME // self.training_interval - BATCH_SIZE // TRAINING_INTERVAL  # 总训练步数预估
        # self.training_step = 0  # 训练步数计数器
        # 构建优化器
        # self._build_optimizer()
        # 新增代码 end -----------------------------------#

    # def _build_optimizer(self):
    #     """ 初始化优化器和调度器 """
    #     # Adam优化器（保持原有参数）
    #     self.optimizer = optim.Adam(
    #         self.model.parameters(),
    #         lr=self.lr,
    #         betas=(0.9, 0.999),
    #         weight_decay=0.0001
    #     )
    #
    #     # 余弦退火调度器
    #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         self.optimizer,
    #         T_max=self.total_steps,  # 总周期步数
    #         eta_min=1e-5  # 最小学习率
    #     )
    #
    #     # 预热调度器
    #     self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #         self.optimizer,
    #         lr_lambda=lambda step: min(step / self.warmup_steps, 1.0)
    #     )

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.Sigmoid()
        )

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        if self.memory_counter > self.batch_size and self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])

        h_train = DM.move_data_to_device(h_train)
        m_train = DM.move_data_to_device(m_train)
        self.model = DM.move_model_to_device(self.model)

        # 检查h_train和m_train是否含有NaN或Inf值
        if torch.isnan(h_train).any().item() or torch.isinf(h_train).any().item():
            raise ValueError("Input tensor h_train contains NaN or Inf values.")
        if torch.isnan(m_train).any().item() or torch.isinf(m_train).any().item():
            raise ValueError("Input tensor m_train contains NaN or Inf values.")

        # train the DNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.9,0.999),weight_decay=0.0001)
        criterion = nn.SmoothL1Loss()

        self.model.train() #设置为训练模式（与之相反的是设置为评估模式：model.eval()）

        optimizer.zero_grad() #梯度清零

        predict = self.model(h_train)
        if torch.isnan(predict).any().item() or torch.isinf(predict).any().item():
            raise ValueError("Model output contains NaN or Inf values.")

        # Ensure m_train is also in the range [0, 1] for BCELoss
        if torch.min(m_train).item() < 0 or torch.max(m_train).item() > 1:
            raise ValueError("m_train values should be in the range [0, 1] for BCELoss")

        loss = criterion(predict, m_train)
        loss.backward()

        # 检查是否存在NaN或Inf的梯度
        has_invalid_gradients = False
        for param in self.model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print("Gradient error: NaN or Inf found!")
                has_invalid_gradients = True

        if not has_invalid_gradients:
            optimizer.step()  # 仅在梯度有效时更新参数

        else:
            print("Skipping this batch due to invalid gradients.")

        self.cost = loss.item()
        assert self.cost >= 0, "Loss must be non-negative"
        self.cost_arr.append(self.cost)

    def decode(self, h, epoch, k = 1, n=0, mode = 'DELTA_KNN'):
        # to have batch dimension when feed into Tensor
        h_tensor = DM.move_data_to_device(torch.Tensor(h[np.newaxis, :]))
        self.model = DM.move_model_to_device(self.model)
        self.model.eval()
        with torch.no_grad():  # 禁用梯度计算，减少内存占用并加速推理
            m_pred = self.model(h_tensor)
        m_pred = m_pred.detach().cpu().numpy()
        # print("预测值为：",m_pred)
        # policies = generate_perturbations(m_pred[0],epoch)
        policies = get_policy_v2(m_pred[0])
        # print("分裂后的预测值为：",policies)
        return policies

    def delta_KNN(self, m, k = 1):
        delta = (np.min(1-m) if np.min(1-m)<np.min(m) else np.min(m)) *0.95 / k
        print("delta=",delta)
        delta_m = []
        for i in range(2*k):
            delta_m.append(m+(i-k)*delta)
        return delta_m

    def decode_test(self, h, k = 1, n=0, mode = 'DELTA_KNN'):
        # to have batch dimension when feed into Tensor
        h_tensor = DM.move_data_to_device(torch.Tensor(h[np.newaxis, :]))
        self.model = DM.move_model_to_device(self.model)
        self.model.eval()
        with torch.no_grad(): #禁用梯度计算，减少内存占用并加速推理
            m_pred = self.model(h_tensor)

        m_pred = m_pred.detach().cpu().numpy() #将预测结果从 GPU 移动到 CPU 并转换为 numpy 数组
        for i in range(len(m_pred[0])):
            if m_pred[0][i] >= 0.98:
                m_pred[0][i] = 1
            if m_pred[0][i] <= 0.02:
                m_pred[0][i] = 0
        # print("预测值为：",m_pred)
        return m_pred[0]

    def plot_cost(self, path,c='b',maker=',',loc=1):
        plot_data(np.arange(len(self.cost_arr))*self.training_interval, self.cost_arr,
                  "loss value",
                  'Time Frames',
                  'Training Loss',
                  path+"loss",c=c,marker=maker,loc=loc)

