# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
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
from utils.dataset_utils import check, separate_data, split_data, save_file, bernoulli_overlap_partition, save_file_with_probs,\
dirichlet_overlap_partition_with_replacement

random.seed(1)
np.random.seed(1)
num_clients = 30
dir_path = "Cifar10_dir_30/"


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
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    if partition == 'bernoulli_overlap':
        seed = 42
        samples_per_client = 5000
        alpha_val = 0.3
        train_data, test_data, train_statistic, test_statistic, class_probs = bernoulli_overlap_partition(
            dataset_image, dataset_label, num_clients, samples_per_client, alpha=alpha_val, seed=seed
        )
        save_file_with_probs(config_path, train_path, test_path, train_data, test_data, num_clients,
                        num_classes, train_statistic, test_statistic, class_probs, seed, 
                        niid, balance, partition)
    elif partition == 'dir_overlap':
        alpha_val = 0.3
        seed = 42
        
        train_data, test_data, train_statistic, test_statistic, class_probs = dirichlet_overlap_partition_with_replacement(
            dataset_image, dataset_label, num_clients, alpha=alpha_val, train_ratio=0.8, seed=seed
        )
        save_file_with_probs(config_path, train_path, test_path, train_data, test_data, num_clients,
                            num_classes, train_statistic, test_statistic, class_probs, seed,
                            niid, balance, partition)

    else:
        X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                        niid, balance, partition, class_per_client=10)
        train_data, test_data = split_data(X, y)
        save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
            statistic, niid, balance, partition)

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)