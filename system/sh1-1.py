import os
import subprocess
import concurrent.futures
import sys
from tqdm import tqdm
import torch  # 需要导入 PyTorch

# # 定义命令列表
# commands = [
#     # PathMNIST_0.1 数据集
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda",
    
#     # PathMNIST_1 数据集
#     "python main.py -data PathMNIST_1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda",
    
#     # PathMNIST_100 数据集
#     "python main.py -data PathMNIST_100 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 9 -dev cuda",
#     "python main.py -data PathMNIST_100 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda"
# ]


# 定义命令列表
# commands = [
#     # PathMNIST_0.1 数据集
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedFomo -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedAMP -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo APFL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedPHP -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedROD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedProto -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedBABU -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedGen -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedALA -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedPAC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedGC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FML -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedKD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedPCL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedCP -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo GPFL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedNTD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedGH -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedDBE -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedCAC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo PFL-DA -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_0.1 -m CNN -algo FedLC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedFomo -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedAMP -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo APFL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedPHP -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedROD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedProto -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedBABU -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedGen -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedALA -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedPAC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedGC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FML -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedKD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedPCL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedCP -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo GPFL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedNTD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedGH -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedDBE -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedCAC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo PFL-DA -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_1 -m CNN -algo FedLC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedFomo -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedAMP -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo APFL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedPHP -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedROD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedProto -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedBABU -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedGen -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedALA -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedPAC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedGC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FML -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedKD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedPCL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedCP -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo GPFL -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedNTD -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedGH -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedDBE -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedCAC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo PFL-DA -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1",
#     "python main.py -data PathMNIST_100 -m CNN -algo FedLC -gr 100 -lr 0.001 -ncl 9 -dev cuda -did 0,1"
# ]


# commands = [
#     # DermaMNIST_0.1 dataset
#     "python main.py -data DermaMNIST_0.1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_0.1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_0.1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_0.1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     # DermaMNIST_1 dataset
#     "python main.py -data DermaMNIST_1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     # DermaMNIST_100 dataset
#     "python main.py -data DermaMNIST_100 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_100 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_100 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_100 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1",
#     "python main.py -data DermaMNIST_100 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 7 -dev cuda -did 0,1"
# ]

# commands = [
#     # OrganAMNIST_0.1 dataset
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     # OrganAMNIST_1 dataset
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     # OrganAMNIST_100 dataset
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1"
# ]

# commands = [
#     # BloodMNIST_0.1 dataset
#     "python main.py -data BloodMNIST_0.1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_0.1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_0.1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_0.1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     # BloodMNIST_1 dataset
#     "python main.py -data BloodMNIST_1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     # BloodMNIST_100 dataset
#     "python main.py -data BloodMNIST_100 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_100 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_100 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_100 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
#     "python main.py -data BloodMNIST_100 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1"
# ]





# # 定义最大并发数
# max_workers = 3

# # 定义一个函数来检查任务是否已经完成
# def is_task_completed(command):
#     # 提取数据集和算法名称
#     data_name = command.split('-data ')[1].split(' ')[0]
#     algo_name = command.split('-algo ')[1].split(' ')[0]
#     output_file = f"{data_name}_{algo_name}.txt"
    
#     # 检查输出文件是否存在且大小大于20KB
#     if os.path.exists(output_file):
#         file_size = os.path.getsize(output_file)
#         return file_size > 20 * 1024  # 20KB
#     return False

# # 定义一个函数来运行单个命令并保存输出
# def run_command(command):
#     # 提取数据集和算法名称
#     data_name = command.split('-data ')[1].split(' ')[0]
#     algo_name = command.split('-algo ')[1].split(' ')[0]
#     output_file = f"{data_name}_{algo_name}.txt"
    
#     # 使用 subprocess 运行命令并将输出保存到文件
#     try:
#         process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         with open(output_file, 'w') as f:
#             while True:
#                 output = process.stdout.readline().decode('utf-8')
#                 if output == '' and process.poll() is not None:
#                     break
#                 if output:
#                     print(output.strip())  # 在控制台中显示
#                     f.write(output)  # 保存到文件
#             # 等待进程结束
#             process.wait()
#         return True
#     except Exception as e:
#         print(f"命令执行出错: {e}")
#         return False

# # 定义一个函数来清理 CUDA 缓存
# def clean_cuda_cache():
#     torch.cuda.empty_cache()
#     print("CUDA 缓存已清理")

# # 使用 ThreadPoolExecutor 来管理线程
# with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#     # 将任务分成批次，每批次最多 max_workers 个任务
#     for i in range(0, len(commands), max_workers):
#         batch_commands = commands[i:i+max_workers]
#         futures = []
#         for cmd in batch_commands:
#             if not is_task_completed(cmd):
#                 futures.append(executor.submit(run_command, cmd))
#             else:
#                 print(f"任务已跳过（已完成）: {cmd}")
        
#         # 等待当前批次的所有任务完成
#         concurrent.futures.wait(futures)
        
#         # 清理 CUDA 缓存
#         clean_cuda_cache()

# print("所有任务已完成！")



# import os
# import subprocess
# import torch
# import concurrent.futures
# import psutil
# from typing import List, Optional
# import time
# import logging

# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class TaskExecutor:
#     def __init__(self, base_workers: int = 3, gpu_memory_threshold: float = 0.9, max_workers_limit: int = 8):
#         self.base_workers = base_workers
#         self.gpu_memory_threshold = gpu_memory_threshold  # GPU内存使用率阈值
#         self.max_workers_limit = max_workers_limit       # 最大并发上限
#         self.active_tasks = {}                           # 跟踪活跃任务及其进程ID
#         self.command_fail_counts = {}                    # 记录每个命令的失败次数

#     def get_gpu_usage(self) -> float:
#         """获取当前GPU内存使用率"""
#         try:
#             if torch.cuda.is_available():
#                 total_memory = torch.cuda.get_device_properties(0).total_memory
#                 used_memory = torch.cuda.memory_reserved(0)
#                 return used_memory / total_memory if total_memory > 0 else 0.0
#             return 0.0
#         except Exception as e:
#             logger.warning(f"获取GPU使用率失败: {e}")
#             return 0.0

#     def adjust_workers(self) -> int:
#         """根据GPU和CPU资源动态调整并发数"""
#         gpu_usage = self.get_gpu_usage()
#         cpu_usage = psutil.cpu_percent() / 100
        
#         if gpu_usage > self.gpu_memory_threshold or cpu_usage > 0.9:
#             return max(1, self.base_workers - 1)  # 减少并发
#         elif gpu_usage < 0.5 and cpu_usage < 0.7:
#             return min(self.max_workers_limit, self.base_workers + 1)  # 增加并发，限制上限
#         return self.base_workers

#     def is_task_completed(self, command: str) -> bool:
#         """检查任务是否已完成"""
#         try:
#             parts = command.split()
#             data_idx = parts.index('-data') + 1
#             algo_idx = parts.index('-algo') + 1
#             data_name = parts[data_idx]
#             algo_name = parts[algo_idx]
#             output_file = f"{data_name}_{algo_name}.txt"
            
#             if os.path.exists(output_file):
#                 file_size = os.path.getsize(output_file)
#                 return file_size > 20 * 1024  # 20KB
#             return False
#         except (ValueError, IndexError) as e:
#             logger.error(f"命令格式错误，无法解析: {command}, 错误: {e}")
#             return False
#         except Exception as e:
#             logger.error(f"检查任务完成状态失败: {e}")
#             return False

#     def run_command(self, command: str) -> bool:
#         """运行单个命令并保存输出"""
#         try:
#             parts = command.split()
#             data_idx = parts.index('-data') + 1
#             algo_idx = parts.index('-algo') + 1
#             data_name = parts[data_idx]
#             algo_name = parts[algo_idx]
#             output_file = f"{data_name}_{algo_name}.txt"
            
#             process = subprocess.Popen(
#                 command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#             )
#             self.active_tasks[command] = process.pid
            
#             with open(output_file, 'w') as f:
#                 while True:
#                     output = process.stdout.readline()
#                     if output == '' and process.poll() is not None:
#                         break
#                     if output:
#                         logger.info(output.strip())
#                         f.write(output)
                
#                 # 捕获并记录错误输出
#                 stderr_output = process.stderr.read()
#                 if stderr_output:
#                     logger.error(f"命令错误输出: {stderr_output}")

#             process.wait()
#             if process.returncode == 0:
#                 del self.active_tasks[command]
#                 torch.cuda.empty_cache()  # 清理当前任务的GPU缓存
#                 return True
#             else:
#                 logger.error(f"命令执行失败，返回码: {process.returncode}")
#                 return False
                
#         except Exception as e:
#             logger.error(f"命令执行出错: {e}")
#             return False
#         finally:
#             if command in self.active_tasks:
#                 del self.active_tasks[command]

#     def execute_tasks(self, commands: List[str], retry_attempts: int = 3) -> None:
#         """执行任务的主函数"""
#         remaining_commands = commands.copy()
        
#         while remaining_commands:
#             max_workers = self.adjust_workers()
#             logger.info(f"当前并发数调整为: {max_workers}")
            
#             try:
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#                     futures = {}
#                     batch_commands = remaining_commands[:max_workers]
                    
#                     for cmd in batch_commands[:]:  # 使用切片避免修改列表时的迭代问题
#                         if not self.is_task_completed(cmd):
#                             # 检查命令是否已经失败两次
#                             if self.command_fail_counts.get(cmd, 0) >= 2:
#                                 logger.warning(f"命令已失败两次，不再尝试: {cmd}")
#                                 remaining_commands.remove(cmd)
#                                 continue
#                             futures[executor.submit(self.run_command, cmd)] = cmd
#                         else:
#                             logger.info(f"任务已跳过（已完成）: {cmd}")
#                             remaining_commands.remove(cmd)
                    
#                     # 处理完成和失败的任务
#                     for future in concurrent.futures.as_completed(futures):
#                         cmd = futures[future]
#                         try:
#                             success = future.result()
#                             if success:
#                                 remaining_commands.remove(cmd)
#                                 # 如果成功，重置失败计数
#                                 if cmd in self.command_fail_counts:
#                                     del self.command_fail_counts[cmd]
#                             else:
#                                 # 记录失败次数
#                                 self.command_fail_counts[cmd] = self.command_fail_counts.get(cmd, 0) + 1
#                                 logger.error(f"命令执行失败，当前失败次数: {self.command_fail_counts[cmd]}")
#                                 if self.command_fail_counts[cmd] >= 2:
#                                     logger.error(f"命令已失败两次，不再尝试: {cmd}")
#                                     remaining_commands.remove(cmd)
#                                 else:
#                                     logger.info(f"重试任务: {cmd} (剩余尝试: {2 - self.command_fail_counts[cmd]})")
#                         except Exception as e:
#                             logger.error(f"任务执行异常: {e}")
                    
#                     time.sleep(1)  # 短暂休眠避免资源竞争
#             except Exception as e:
#                 logger.error(f"线程池执行出错: {e}")
#                 time.sleep(5)  # 出错后等待更长时间再重试
        
#         logger.info("所有任务已完成！")

        
# # 使用示例
# if __name__ == "__main__":
# 定义命令列表
    # commands = [
    # # BloodMNIST_0.1 数据集
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedFomo -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedAMP -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo APFL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedPHP -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedROD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedProto -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedBABU -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedGen -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedALA -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedPAC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedGC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FML -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedKD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedPCL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedCP -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo GPFL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedNTD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedGH -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedDBE -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedCAC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo PFL-DA -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_0.1 -m CNN -algo FedLC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # # BloodMNIST_1 数据集
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedFomo -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedAMP -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo APFL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedPHP -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedROD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedProto -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedBABU -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedGen -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedALA -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedPAC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedGC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FML -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedKD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedPCL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedCP -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo GPFL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedNTD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedGH -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedDBE -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedCAC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo PFL-DA -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_1 -m CNN -algo FedLC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # # BloodMNIST_100 数据集
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedFomo -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedAMP -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo APFL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedPHP -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedROD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedProto -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedBABU -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedGen -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedALA -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedPAC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedGC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FML -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedKD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedPCL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedCP -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo GPFL -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedNTD -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedGH -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedDBE -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedCAC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo PFL-DA -gr 100 -lr 0.001 -ncl 8 -dwev cuda -did 0,1",
    # "python main.py -data BloodMNIST_100 -m CNN -algo FedLC -gr 100 -lr 0.001 -ncl 8 -dev cuda -did 0,1"
    # ]
#    commands = [
#     # OrganAMNIST_0.1 数据集
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedFomo -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedAMP -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo APFL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedPHP -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedROD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedProto -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedBABU -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedGen -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedALA -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedPAC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedGC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FML -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedKD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedPCL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedCP -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo GPFL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedNTD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedGH -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedDBE -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedCAC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo PFL-DA -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_0.1 -m CNN -algo FedLC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     # OrganAMNIST_1 数据集
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedFomo -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedAMP -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo APFL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedPHP -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedROD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedProto -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedBABU -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedGen -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedALA -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedPAC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedGC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FML -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedKD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedPCL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedCP -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo GPFL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedNTD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedGH -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedDBE -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedCAC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo PFL-DA -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_1 -m CNN -algo FedLC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     # OrganAMNIST_100 数据集
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedMTL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo PerAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo pFedMe -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedFomo -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedAMP -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo APFL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedPer -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedPHP -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedBN -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedROD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedProto -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedDyn -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedBABU -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo APPLE -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedGen -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo SCAFFOLD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedALA -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedPAC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedGC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FML -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedKD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedPCL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedCP -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo GPFL -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedNTD -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedGH -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedDBE -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedCAC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo PFL-DA -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
#     "python main.py -data OrganAMNIST_100 -m CNN -algo FedLC -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1"
#     ]

    # commands = [
    #     # OrganAMNIST_0.1 数据集
    #     "python main1.py -data OrganAMNIST_0.1 -m CNN -algo FedAvg1 -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_0.1 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_0.1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_0.1 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_0.1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_0.1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_0.1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # OrganAMNIST_1 数据集
    #     "python main1.py -data OrganAMNIST_1 -m CNN -algo FedAvg1 -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_1 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_1 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_1 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_1 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_1 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_1 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # OrganAMNIST_100 数据集
    #     "python main1.py -data OrganAMNIST_100 -m CNN -algo FedAvg1 -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_100 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_100 -m CNN -algo FedDistill -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_100 -m CNN -algo FedProx -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_100 -m CNN -algo FedRep -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_100 -m CNN -algo LG-FedAvg -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_100 -m CNN -algo Ditto -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1",
    #     # "python main1.py -data OrganAMNIST_100 -m CNN -algo MOON -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1"
    # ]
    # executor = TaskExecutor(base_workers=3, max_workers_limit=8)
    # executor.execute_tasks(commands)

# import os
# import subprocess
# import torch
# import concurrent.futures
# import psutil
# from typing import List, Optional
# import time
# import logging

# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class TaskExecutor:
#     def __init__(self, base_workers: int = 3, gpu_memory_reserved: float = 0.3, max_workers_limit: int = 2):
#         self.base_workers = base_workers
#         self.gpu_memory_reserved = gpu_memory_reserved  # 保留30%的GPU内存
#         self.max_workers_limit = max_workers_limit     # 最大并发上限
#         self.active_tasks = {}                         # 跟踪活跃任务及其进程ID
#         self.command_fail_counts = {}                  # 记录每个命令的失败次数
#         self.avg_task_time = 1.0                      # 初始平均任务时间

#     def get_system_resources(self) -> dict:
#         """获取系统资源使用情况"""
#         try:
#             if torch.cuda.is_available():
#                 total_memory = torch.cuda.get_device_properties(0).total_memory
#                 used_memory = torch.cuda.memory_reserved(0)
#                 gpu_usage = used_memory / total_memory if total_memory > 0 else 0.0
#                 gpu_available = total_memory * (1 - gpu_usage)
#             else:
#                 gpu_usage, gpu_available = 0.0, 0.0
#         except Exception as e:
#             logger.warning(f"获取GPU使用率失败: {e}")
#             gpu_usage, gpu_available = 0.0, 0.0

#         cpu_usage = psutil.cpu_percent() / 100
#         memory = psutil.virtual_memory()
#         return {
#             "gpu_usage": gpu_usage,
#             "gpu_available": gpu_available,
#             "cpu_usage": cpu_usage,
#             "memory_usage": memory.percent / 100,
#             "memory_available": memory.available
#         }

#     def estimate_max_workers(self) -> int:
#         """根据系统资源预估最大并发数"""
#         resources = self.get_system_resources()
#         memory_per_task = 2 * 1024 * 1024 * 1024  # 2GB
        
#         gpu_based_workers = int(resources["gpu_available"] * (1 - self.gpu_memory_reserved) / memory_per_task)
#         ram_based_workers = int(resources["memory_available"] * 0.7 / memory_per_task)  # 保留30% RAM
        
#         if resources["gpu_usage"] > 0.9 or resources["cpu_usage"] > 0.9 or resources["memory_usage"] > 0.9:
#             return max(1, self.base_workers - 1)
#         return min(self.max_workers_limit, max(self.base_workers, min(gpu_based_workers, ram_based_workers)))

#     def is_task_completed(self, command: str) -> bool:
#         """检查任务是否已完成"""
#         try:
#             parts = command.split()
#             data_idx = parts.index('-data') + 1
#             algo_idx = parts.index('-algo') + 1
#             data_name = parts[data_idx]
#             algo_name = parts[algo_idx]
#             output_file = f"{data_name}_{algo_name}.txt"
            
#             if os.path.exists(output_file):
#                 file_size = os.path.getsize(output_file)
#                 return file_size > 20 * 1024  # 20KB
#             return False
#         except (ValueError, IndexError) as e:
#             logger.error(f"命令格式错误，无法解析: {command}, 错误: {e}")
#             return False
#         except Exception as e:
#             logger.error(f"检查任务完成状态失败: {e}")
#             return False

#     def run_command(self, command: str) -> bool:
#         """运行单个命令并保存输出"""
#         try:
#             parts = command.split()
#             data_idx = parts.index('-data') + 1
#             algo_idx = parts.index('-algo') + 1
#             data_name = parts[data_idx]
#             algo_name = parts[algo_idx]
#             output_file = f"{data_name}_{algo_name}.txt"
            
#             process = subprocess.Popen(
#                 command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#             )
#             self.active_tasks[command] = process.pid
            
#             with open(output_file, 'w') as f:
#                 while True:
#                     output = process.stdout.readline()
#                     if output == '' and process.poll() is not None:
#                         break
#                     if output:
#                         logger.info(output.strip())
#                         f.write(output)
                
#                 stderr_output = process.stderr.read()
#                 if stderr_output:
#                     logger.error(f"命令错误输出: {stderr_output}")

#             process.wait()
#             if process.returncode == 0:
#                 del self.active_tasks[command]
#                 torch.cuda.empty_cache()
#                 return True
#             else:
#                 logger.error(f"命令执行失败，返回码: {process.returncode}")
#                 return False
                
#         except Exception as e:
#             logger.error(f"命令执行出错: {e}")
#             return False
#         finally:
#             if command in self.active_tasks:
#                 del self.active_tasks[command]

#     def execute_tasks(self, commands: List[str], retry_attempts: int = 3) -> None:
#         """执行任务的主函数"""
#         remaining_commands = commands.copy()
        
#         while remaining_commands:
#             start_time = time.time()
#             max_workers = self.estimate_max_workers()
#             resources = self.get_system_resources()
#             logger.info(f"当前并发数: {max_workers}, 资源: GPU={resources['gpu_usage']:.2%}, CPU={resources['cpu_usage']:.2%}, RAM={resources['memory_usage']:.2%}")
            
#             try:
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#                     futures = {}
#                     # 按失败次数排序，优先执行失败少的任务
#                     batch_commands = sorted(
#                         remaining_commands[:max_workers],
#                         key=lambda x: self.command_fail_counts.get(x, 0)
#                     )
                    
#                     for cmd in batch_commands[:]:
#                         if not self.is_task_completed(cmd):
#                             if self.command_fail_counts.get(cmd, 0) >= 2:
#                                 logger.warning(f"命令已失败两次，不再尝试: {cmd}")
#                                 remaining_commands.remove(cmd)
#                                 continue
#                             futures[executor.submit(self.run_command, cmd)] = cmd
#                         else:
#                             logger.info(f"任务已跳过（已完成）: {cmd}")
#                             remaining_commands.remove(cmd)
                    
#                     for future in concurrent.futures.as_completed(futures):
#                         cmd = futures[future]
#                         try:
#                             success = future.result()
#                             if success:
#                                 remaining_commands.remove(cmd)
#                                 if cmd in self.command_fail_counts:
#                                     del self.command_fail_counts[cmd]
#                             else:
#                                 self.command_fail_counts[cmd] = self.command_fail_counts.get(cmd, 0) + 1
#                                 if self.command_fail_counts[cmd] >= 2:
#                                     logger.error(f"命令已失败两次，不再尝试: {cmd}")
#                                     remaining_commands.remove(cmd)
#                                 else:
#                                     wait_time = 2 ** self.command_fail_counts[cmd]  # 指数退避
#                                     logger.info(f"任务失败，将在 {wait_time}s 后重试: {cmd}")
#                                     time.sleep(wait_time)
#                         except Exception as e:
#                             logger.error(f"任务执行异常: {e}")
                    
#                     # 更新平均任务时间并动态调整休眠
#                     batch_time = time.time() - start_time
#                     self.avg_task_time = (self.avg_task_time * 0.9 + batch_time * 0.1)
#                     sleep_time = min(5.0, max(0.5, self.avg_task_time * 0.1))
#                     logger.info(f"批次执行时间: {batch_time:.2f}s, 下次休眠: {sleep_time:.2f}s")
#                     time.sleep(sleep_time)
#             except Exception as e:
#                 logger.error(f"线程池执行出错: {e}")
#                 time.sleep(5)
        
#         logger.info("所有任务已完成！")

# if __name__ == "__main__":
#     datasets = ["BloodMNIST_0.1", "BloodMNIST_1", "BloodMNIST_100",
#                 "DermaMNIST_0.1", "DermaMNIST_1", "DermaMNIST_100",
#                 "OrganAMNIST_0.1", "OrganAMNIST_1", "OrganAMNIST_100"]
#     algorithms = [
#     "FedAvg", "Local", "FedMTL", "PerAvg", "pFedMe", "FedProx", "FedFomo", "FedAMP",
#     "APFL", "FedPer", "Ditto", "FedRep", "FedPHP", "FedBN", "FedROD", "FedProto",
#     "FedDyn", "MOON", "FedBABU", "APPLE", "FedGen", "SCAFFOLD", "FedDistill",
#     "FedALA", "FedPAC", "LG-FedAvg", "FedGC", "FML", "FedKD", "FedPCL", "FedCP",
#     "GPFL", "FedNTD", "FedGH", "FedDBE", "FedCAC", "PFL-DA", "FedLC"
#     ]    
#     # Define number of classes for each dataset type
#     class_numbers = {
#         "BloodMNIST": 8,   # 8 classes
#         "DermaMNIST": 7,   # 7 classes
#         "OrganAMNIST": 11  # 11 classes
#     }

#     commands = []
#     for dataset in datasets:
#         # Extract the base dataset name (without suffix)
#         base_dataset = dataset.split("_")[0]
#         ncl = class_numbers[base_dataset]  # Get the number of classes
        
#         for algo in algorithms:
#             cmd = f"python main1.py -data {dataset} -m CNN -algo {algo} -gr 100 -lr 0.001 -ncl {ncl} -dev cuda -did 0,1"
#             commands.append(cmd)
    
#     executor = TaskExecutor(gpu_memory_reserved=0.3)
#     executor.execute_tasks(commands)


import os
import subprocess
import torch
import concurrent.futures
import psutil
from typing import List
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskExecutor:
    def __init__(self, base_workers: int = 3, gpu_memory_threshold: float = 0.9, max_workers_limit: int = 8):
        self.base_workers = base_workers
        self.gpu_memory_threshold = gpu_memory_threshold  # GPU内存使用率阈值
        self.max_workers_limit = max_workers_limit       # 最大并发上限
        self.active_tasks = {}                           # 跟踪活跃任务及其进程ID
        self.command_fail_counts = {}                    # 记录每个命令的失败次数

    def get_gpu_usage(self) -> float:
        """获取当前GPU内存使用率"""
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                used_memory = torch.cuda.memory_reserved(0)
                return used_memory / total_memory if total_memory > 0 else 0.0
            return 0.0
        except Exception as e:
            logger.warning(f"获取GPU使用率失败: {e}")
            return 0.0

    def adjust_workers(self) -> int:
        """根据GPU和CPU资源动态调整并发数"""
        gpu_usage = self.get_gpu_usage()
        cpu_usage = psutil.cpu_percent() / 100
        
        if gpu_usage > self.gpu_memory_threshold or cpu_usage > 0.9:
            return max(1, self.base_workers - 1)  # 减少并发
        elif gpu_usage < 0.5 and cpu_usage < 0.7:
            return min(self.max_workers_limit, self.base_workers + 1)  # 增加并发，限制上限
        return self.base_workers

    def is_task_completed(self, command: str) -> bool:
        """检查任务是否已完成"""
        try:
            parts = command.split()
            data_idx = parts.index('-data') + 1
            algo_idx = parts.index('-algo') + 1
            data_name = parts[data_idx]
            algo_name = parts[algo_idx]
            output_file = f"{data_name}_{algo_name}.txt"
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                return file_size > 20 * 1024  # 20KB
            return False
        except (ValueError, IndexError) as e:
            logger.error(f"命令格式错误，无法解析: {command}, 错误: {e}")
            return False
        except Exception as e:
            logger.error(f"检查任务完成状态失败: {e}")
            return False

    def run_command(self, command: str) -> bool:
        """运行单个命令并保存输出"""
        try:
            parts = command.split()
            data_idx = parts.index('-data') + 1
            algo_idx = parts.index('-algo') + 1
            data_name = parts[data_idx]
            algo_name = parts[algo_idx]
            output_file = f"{data_name}_{algo_name}.txt"
            
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            self.active_tasks[command] = process.pid
            
            with open(output_file, 'w') as f:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        logger.info(output.strip())
                        f.write(output)
                
                stderr_output = process.stderr.read()
                if stderr_output:
                    logger.error(f"命令错误输出: {stderr_output}")

            process.wait()
            if process.returncode == 0:
                del self.active_tasks[command]
                torch.cuda.empty_cache()
                return True
            else:
                logger.error(f"命令执行失败，返回码: {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"命令执行出错: {e}")
            return False
        finally:
            if command in self.active_tasks:
                del self.active_tasks[command]

    def execute_tasks(self, commands: List[str], retry_attempts: int = 3) -> None:
        """执行任务的主函数"""
        remaining_commands = commands.copy()
        
        while remaining_commands:
            max_workers = self.adjust_workers()
            logger.info(f"当前并发数调整为: {max_workers}")
            
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    batch_commands = remaining_commands[:max_workers]
                    
                    for cmd in batch_commands[:]:
                        if not self.is_task_completed(cmd):
                            if self.command_fail_counts.get(cmd, 0) >= 2:
                                logger.warning(f"命令已失败两次，不再尝试: {cmd}")
                                remaining_commands.remove(cmd)
                                continue
                            futures[executor.submit(self.run_command, cmd)] = cmd
                        else:
                            logger.info(f"任务已跳过（已完成）: {cmd}")
                            remaining_commands.remove(cmd)
                    
                    for future in concurrent.futures.as_completed(futures):
                        cmd = futures[future]
                        try:
                            success = future.result()
                            if success:
                                remaining_commands.remove(cmd)
                                if cmd in self.command_fail_counts:
                                    del self.command_fail_counts[cmd]
                            else:
                                self.command_fail_counts[cmd] = self.command_fail_counts.get(cmd, 0) + 1
                                logger.error(f"命令执行失败，当前失败次数: {self.command_fail_counts[cmd]}")
                                if self.command_fail_counts[cmd] >= 2:
                                    logger.error(f"命令已失败两次，不再尝试: {cmd}")
                                    remaining_commands.remove(cmd)
                                else:
                                    logger.info(f"重试任务: {cmd} (剩余尝试: {2 - self.command_fail_counts[cmd]})")
                        except Exception as e:
                            logger.error(f"任务执行异常: {e}")
                    
                    time.sleep(1)
            except Exception as e:
                logger.error(f"线程池执行出错: {e}")
                time.sleep(5)
        
        logger.info("所有任务已完成！")

if __name__ == "__main__":
    datasets = ["OrganAMNIST_0.1", "OrganAMNIST_1", "OrganAMNIST_100" , "BloodMNIST_1", "BloodMNIST_100",
                "DermaMNIST_0.1", "DermaMNIST_1", "DermaMNIST_100",
                "OrganAMNIST_0.1", "OrganAMNIST_1", "OrganAMNIST_100"] 
    algorithms = [
        "FedAvg", "FedProx","FedBN", "MOON", "FedBABU","FedGen","FedNTD", "FedDBE",  "FedLC", "FedAvg_PIFCA"
    ]
    class_numbers = {
        "BloodMNIST": 8,
        "DermaMNIST": 7,
        "OrganAMNIST": 11
    }

    commands = []
    for dataset in datasets:
        base_dataset = dataset.split("_")[0]
        ncl = class_numbers[base_dataset]
        for algo in algorithms:
            cmd = f"python main1.py -data {dataset} -m CNN -algo {algo} -gr 100 -lr 0.001 -ncl {ncl} -dev cuda -did 0,1"
            commands.append(cmd)
    
    executor = TaskExecutor(base_workers=3, max_workers_limit=8)
    executor.execute_tasks(commands)