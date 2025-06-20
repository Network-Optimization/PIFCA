# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 10
# --- MODIFIED: 更新目录路径以反映SVHN数据集 ---
dir_path = "SVHN_100_nc10/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # --- MODIFIED: 调整transform以适应SVHN的3通道彩色图像 ---
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # --- MODIFIED: 获取 SVHN 数据 ---
    # 加载 SVHN 训练集
    trainset = torchvision.datasets.SVHN(
        root=dir_path+"rawdata", split='train', download=True, transform=transform)
    # 加载 SVHN 测试集
    testset = torchvision.datasets.SVHN(
        root=dir_path+"rawdata", split='test', download=True, transform=transform)
    
    # --- MODIFIED: 采用更清晰的方式加载和合并数据 ---
    # 创建DataLoader以一次性加载所有数据（保持原代码逻辑）
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    # 从DataLoader中提取数据和标签
    for _, (train_images, train_labels) in enumerate(trainloader, 0):
        pass  # 循环结束后，变量将持有整个数据集的张量
    for _, (test_images, test_labels) in enumerate(testloader, 0):
        pass

    # 将训练集和测试集合并，然后转换为numpy数组
    dataset_image = torch.cat((train_images, test_images), dim=0).numpy()
    dataset_label = torch.cat((train_labels, test_labels), dim=0).numpy()
    # 注意: torchvision的SVHN数据集自动将标签10转换为0，无需手动处理

    num_classes = len(np.unique(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                      niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)