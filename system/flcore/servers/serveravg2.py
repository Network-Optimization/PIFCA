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

import time
from flcore.clients.clientavg1 import clientAVG1
from flcore.servers.serverbase import Server
from threading import Thread
import random

class FedAvg2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG1)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")


        # user_indices = list(range(self.num_clients))

        # # 随机打乱用户顺序
        # random.shuffle(user_indices)

        # # 确保每个簇至少有1个用户
        # for i in range(self.num_clu):
        #     self.clusters[i].append(user_indices[i])

        # # 剩余用户随机分配到任意簇
        # for i in range(self.num_clu, self.num_clients):
        #     cluster_id = random.randint(0, self.num_clu - 1)
        #     self.clusters[cluster_id].append(user_indices[i])
        # for cluster_id, client_ids in enumerate(self.clusters):
        #     for client_id in client_ids:
        #         self.clients[client_id].clu_id = cluster_id  #为客户端分配簇类编号

        # self.load_model()
        self.Budget = []


    def train(self):
        cluster_stable_counter = 0  # 记录聚类稳定的轮次数
        last_cluster_assignment = None  # 记录上一轮的聚类分配
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            # ========== 关键修改点 ========== 
            # 每轮都必须发送全局模型（标准FedAvg要求）
            # 移出if条件，保证每轮执行
            
            # 只有聚类稳定后才额外发送聚类模型
            if cluster_stable_counter >= 3:  
                self.send_clu_models()
            else:
                self.send_models()


            # 评估逻辑
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # 客户端训练
            for client in self.selected_clients:
                client.train()
           
            if cluster_stable_counter != 3:  
                current_clusters = self.check_cluster_stability()
                
                if last_cluster_assignment is not None:
                    if current_clusters == last_cluster_assignment:
                        cluster_stable_counter += 1
                        print(f"Cluster stability count: {cluster_stable_counter}/5")
                    else:
                        cluster_stable_counter = 0  # 重置计数器
                        print("Cluster assignment changed, reset stability counter")

            if cluster_stable_counter >= 3:
                self.receive_clu_models()
                self.aggregate_cluster_models()
            else:
                self.receive_models()
                self.aggregate_parameters()
            
                
                last_cluster_assignment = current_clusters
            
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
            # self.print_(max(self.rs_test_acc), max(
            #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG1)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
