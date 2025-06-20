import numpy as np
import os
import sys
import random
import torch
from medmnist import BloodMNIST, INFO  # 修改为 BloodMNIST
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 10
dir_path = "BloodMNIST_100/"  # 修改目录名为 BloodMNIST_100

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

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),                        # 转换为张量并归一化到 [0, 1]
        transforms.Normalize([0.5], [0.5])            # 归一化到 [-1, 1]
    ])

    # 加载训练集和测试集
    train_dataset = BloodMNIST(  # 修改为 BloodMNIST
        root=dir_path + "rawdata",
        split="train",
        download=True,
        transform=transform
    )
    test_dataset = BloodMNIST(  # 修改为 BloodMNIST
        root=dir_path + "rawdata",
        split="test",
        download=True,
        transform=transform
    )

    # 获取数据和标签
    train_images = train_dataset.imgs
    train_labels = train_dataset.labels
    test_images = test_dataset.imgs
    test_labels = test_dataset.labels

    # 只使用十分之一的数据
    train_size = len(train_images)
    test_size = len(test_images)
    train_indices = np.random.choice(train_size, train_size // 1, replace=False)
    test_indices = np.random.choice(test_size, test_size // 1, replace=False)

    train_images = train_images[train_indices]
    train_labels = train_labels[train_indices]
    test_images = test_images[test_indices]
    test_labels = test_labels[test_indices]

    # 合并训练集和测试集
    dataset_image = np.concatenate([train_images, test_images], axis=0)
    dataset_label = np.concatenate([train_labels, test_labels], axis=0)

    if len(dataset_label.shape) > 1:
        dataset_label = dataset_label.flatten()

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # BloodMNIST 是单通道灰度图像，形状为 (H, W)，需要转换为 (C, H, W)
    if len(dataset_image.shape) == 1:  # 如果是 (N, H, W)
        dataset_image = np.expand_dims(dataset_image, axis=1)  # 转换为 (N, C, H, W), C=1

    # 数据分配
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