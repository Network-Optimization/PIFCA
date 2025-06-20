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
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
import random
import copy
import torch
import numpy as np
import torch.nn.functional as F
import math
from sklearn.decomposition import PCA
from threading import Thread

class Selfatten(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.L=[]


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if i ==0:
                self.send_models()
            else:
                self.send_attenmodels(self.L)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            z, p = self.pca_compress_models(self.clients)
            A = self.compute_attention_matrix_fixed(z)
            self.L = []
            for i in self.clients:
                self.L.append(copy.deepcopy(i.model))
            for k in self.L:
                for a in k.parameters():
                    a.data.zero_()
            for x, i in enumerate(self.L):
                for y, j in enumerate(self.clients):
                    for a, b in zip(i.parameters(), j.model.parameters()):
                        a.data += A[x][y]*b.data
            # self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            # self.aggregate_parameters()

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
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        
    def flatten_model(self,client):
        """
        将 PyTorch 模型的所有参数 flatten 并拼接成一个 1D 向量
        """
        with torch.no_grad():
            return torch.cat([p.view(-1).cpu() for p in client.model.parameters()]).numpy()  # shape: [D]

    def pca_compress_models(self,model_list, n_components=10):
        """
        输入多个模型，将它们的参数 flatten 后使用 PCA 降维
        返回每个模型对应的嵌入向量 z_i ∈ ℝ^128
        """
        # 提取参数向量矩阵：X ∈ [K, D]
        param_matrix = np.stack([self.flatten_model(client) for client in model_list])  # shape: [K, D]

        # 执行 PCA 降维：Z ∈ [K, n_components]
        pca = PCA(n_components=n_components,  svd_solver='randomized')
        z_matrix = pca.fit_transform(param_matrix)  # 每行是一个模型的嵌入向量 z_i

        return z_matrix, pca  # 返回嵌入向量 + PCA 模型（可用于新模型 transform）
        
    def compute_attention_matrix_fixed(self, Z, qk_dim=64, seed=42):
        """
        Z: torch.Tensor, shape [K, d]，每个客户端的嵌入向量
        qk_dim: 投影维度
        返回 A: torch.Tensor, shape [K, K] 注意力矩阵
        """
        torch.manual_seed(seed)  # 保证每次结果一致
        Z_tensor = torch.tensor(Z, dtype=torch.float32)
      

        # 随机初始化 Q, K 权重矩阵（不训练）
        W_Q = torch.randn(Z.shape[1], qk_dim) / math.sqrt(Z.shape[1])
        W_K = torch.randn(Z.shape[1], qk_dim) / math.sqrt(Z.shape[1])

        Q = Z_tensor @ W_Q  # 成功

        K_ = Z_tensor @ W_K  # [K, qk_dim]

        scores = Q @ K_.T / math.sqrt(qk_dim)  # [K, K]
        A = F.softmax(scores, dim=1)  # 行归一化，注意力矩阵
        return A
