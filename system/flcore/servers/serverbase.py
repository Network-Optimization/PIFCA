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

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import torch.nn.functional as F
from utils.data_utils import read_client_data
from utils.dlg import DLG
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity



class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        # self.num_clu = 3
        # self.cluster_uploads = {
        #     cluster_id: {"ids": [], "weights": [], "models": [], "tot_samples": 0}
        #     for cluster_id in range(self.num_clu)
        # }
        # self.clusters_model = [copy.deepcopy(self.global_model) for _ in range(self.num_clu)] #存放模型
        # self.clusters = [{} for _ in range(self.num_clu)]  # 存放编号
        self.stable = 0

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    
    def set_clients_clu(self, clientObj, client_idx):
        
        for i, train_slow, send_slow in zip(client_idx, self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
    def set_clients_clu(self, clientObj, client_idx):
        
        for i, train_slow, send_slow in zip(client_idx, self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # def send_clu_models(self):
    #     for id, i in enumerate(self.clusters):
    #         for a in i:
    #             start_time = time.time()

    #             self.clients[a].set_parameters(self.clusters_model[id])
    #             print(f"Send: client ID {self.clients[a].id}, clusters ID {self.clients[a].clu_id}")
    #             self.clients[a].send_time_cost['num_rounds'] += 1
    #             self.clients[a].send_time_cost['total_cost'] = 2 * (time.time() - start_time)
    def send_clu_models(self):
        for cluster_label, client_indices in self.clusters.items():  # 直接遍历字典的键值对
            for client_idx in client_indices:
                start_time = time.time()
                # 设置参数
                self.clients[client_idx].set_parameters(self.clusters_model[cluster_label])
                print(f"Send: client ID {self.clients[client_idx].id}, cluster ID {cluster_label}")
                # 记录时间
                self.clients[client_idx].send_time_cost['num_rounds'] += 1
                self.clients[client_idx].send_time_cost['total_cost'] = 2 * (time.time() - start_time)

    def send_clu_models_test(self):
        
            start_time = time.time()
            # 设置参数
            for i in self.clients:
                i.set_parameters(self.clusters_model[i.clu_id])
                print(f"Send: client ID {i.id}, cluster ID {i.clu_id}")
                # 记录时间
                i.send_time_cost['num_rounds'] += 1
                i.send_time_cost['total_cost'] = 2 * (time.time() - start_time)
        

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def receive_clu_models(self):
        """接收客户端模型，并按簇分类存储"""
        assert len(self.selected_clients) > 0, "无客户端被选中"

        # 筛选活跃客户端（考虑dropout）
        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )

        # 分类存储模型
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] + client.send_time_cost['total_cost']
                # print("client {} train time : {}".format(client.id, client.train_time_cost['total_cost']))
                # print("client {} communication time : {}".format(client.id, client.send_time_cost['total_cost']))
                # print("client {} total time : {}".format(client.id, client_time_cost))
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                cluster_id = client.clu_id  # 
                self.cluster_uploads[cluster_id]["ids"].append(client.id)
                self.cluster_uploads[cluster_id]["weights"].append(client.train_samples)
                self.cluster_uploads[cluster_id]["models"].append(client.model)
                self.cluster_uploads[cluster_id]["tot_samples"] += client.train_samples

        # 计算簇内权重（归一化）
        for cluster_id in self.cluster_uploads:
            tot_samples = self.cluster_uploads[cluster_id]["tot_samples"]
            if tot_samples > 0:
                self.cluster_uploads[cluster_id]["weights"] = [
                    w / tot_samples for w in self.cluster_uploads[cluster_id]["weights"]
                ]
                
    def clu_models(self):
        """接收客户端模型，并按簇分类存储"""
        assert len(self.selected_clients) > 0, "无客户端被选中"

        # 筛选活跃客户端（考虑dropout）
        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )

        # 分类存储模型
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] + client.send_time_cost['total_cost']
                # print("client {} train time : {}".format(client.id, client.train_time_cost['total_cost']))
                # print("client {} communication time : {}".format(client.id, client.send_time_cost['total_cost']))
                # print("client {} total time : {}".format(client.id, client_time_cost))
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                cluster_id = client.clu_id  # 
                for agg_param, client_param in zip(client.model.parameters(), self.global_model.parameters()):
                    agg_param.data -= client_param.data 
                self.cluster_uploads[cluster_id]["ids"].append(client.id)
                self.cluster_uploads[cluster_id]["weights"].append(client.train_samples)
                self.cluster_uploads[cluster_id]["models"].append(client.model)
                self.cluster_uploads[cluster_id]["tot_samples"] += client.train_samples
    def cluster_clients_by_gradient(self, clients, global_model, num_clusters):

    # 1. 收集所有客户端的梯度（模型与全局模型的差值）
        gradients = []
        for i, client in enumerate(clients):
            client_grad = []
            for c_param, g_param in zip(client.model.parameters(), global_model.parameters()):
                client_grad.append((c_param.data - g_param.data).cpu().numpy())
            gradients.append(np.concatenate([g.flatten() for g in client_grad]))
        gradients = np.array(gradients)  # 转换为NumPy矩阵 [num_clients, grad_dim]

        # 2. 初始化：每个客户端一个簇
        clusters = [{i} for i in range(len(clients))]
        cluster_grads = [gradients[i] for i in range(len(clients))]

        # 3. 层次聚类（直到达到目标簇数）
        while len(clusters) > num_clusters:
            best_gain = -np.inf
            best_pair = (0, 1)  # 默认选择第一对

            # 遍历所有可能的簇对组合
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # 计算合并后的梯度范数平方增益
                    combined = cluster_grads[i] + cluster_grads[j]
                    gain = np.linalg.norm(combined)**2 - (np.linalg.norm(cluster_grads[i])**2 + np.linalg.norm(cluster_grads[j])**2)
                    
                    # 更新最佳增益对
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, j)

            # 合并最佳簇对
            i, j = best_pair
            clusters[i].update(clusters[j])
            cluster_grads[i] += cluster_grads[j]
            del clusters[j]
            del cluster_grads[j]
        for cluster_id, member_indices in enumerate(clusters):
            for idx in member_indices:
                clients[idx].clu_id = cluster_id
        for i in clients:
            print(f'client ID: {i.id}, cluster ID: {i.clu_id}')
        self.clusters = copy.deepcopy(clusters)

    def cluster_clients_by_gradients(self, clients, global_model):

        gradients = []
        for client in clients:
            client_grad = []
            for c_param, g_param in zip(client.model.parameters(), global_model.parameters()):
                client_grad.append((c_param.data - g_param.data).cpu().numpy())
            gradients.append(np.concatenate([g.flatten() for g in client_grad]))
      

        self.stable = 1
        gradients = np.array(gradients)  # 转换为NumPy矩阵 [num_clients, grad_dim]

        similarity_matrix = - cosine_similarity(gradients)
        
        # 亲和传播
        # 将相似度矩阵转换为适合AP算法的偏好值
        # 这里使用中位数作为偏好值，可以根据需要调整
        preference = np.median(similarity_matrix)
        af = AffinityPropagation(affinity='precomputed', 
                            preference=preference,
                            random_state=42).fit(similarity_matrix)
        cluster_labels = af.labels_
        for idx, client in enumerate(clients):
            client.clu_id = int(cluster_labels[idx])

        for client in clients:
            print(f'Client ID: {client.id}, Cluster ID: {client.clu_id}')

        self.clusters = {}
        for label in set(cluster_labels):
            self.clusters[label] = [client.id for client in clients if client.clu_id == label]
        self.num_clu = len(set(cluster_labels))
        self.cluster_uploads = {cluster_id: {"ids": [], "weights": [], "models": [], "tot_samples": 0}
        for cluster_id in range(self.num_clu)
        }
        self.clusters_model = [copy.deepcopy(self.global_model) for _ in range(self.num_clu)]


        

    def cluster_test(self):

        # gradients = []
        # for client in clients:
        #     client_grad = []
        #     for c_param, g_param in zip(client.model.parameters(), global_model.parameters()):
        #         client_grad.append((c_param.data - g_param.data).cpu().numpy())
        #     gradients.append(np.concatenate([g.flatten() for g in client_grad]))
        # grad_norms = [np.linalg.norm(grad) for grad in gradients]
        # max_norm = np.max(grad_norms)
        # mean_norm = np.mean(grad_norms)    
        # # if (max_norm-mean_norm)/(mean_norm) >= 0.3 and self.stable == 0:
        # self.stable = 1
        # gradients = np.array(gradients)  # 转换为NumPy矩阵 [num_clients, grad_dim]

        # similarity_matrix = cosine_similarity(gradients)
    
        # clustering = AgglomerativeClustering( metric="precomputed", linkage="complete").fit(-similarity_matrix) 
        # cluster_labels = clustering.labels_
        # for idx, client in enumerate(clients):
        #     client.clu_id = int(cluster_labels[idx])

        # for client in clients:
        #     print(f'Client ID: {client.id}, Cluster ID: {client.clu_id}')

        # self.clusters = {}
        # for label in set(cluster_labels):
        #     self.clusters[label] = [client.id for client in clients if client.clu_id == label]
        self.num_clu = 4
        self.cluster_uploads = {cluster_id: {"ids": [], "weights": [], "models": [], "tot_samples": 0}
        for cluster_id in range(self.num_clu)
        }
        self.clusters_model = [copy.deepcopy(self.global_model) for _ in range(self.num_clu)]

        # else:
        #     pass

    def cluster_avg3(self, clients, global_model):

        gradients = []
        for client in clients:
            client_grad = []
            for c_param, g_param in zip(client.model.parameters(), global_model.parameters()):
                client_grad.append((c_param.data - g_param.data).cpu().numpy())
            gradients.append(np.concatenate([g.flatten() for g in client_grad]))
        grad_norms = [np.linalg.norm(grad) for grad in gradients]
        max_norm = np.max(grad_norms)
        mean_norm = np.mean(grad_norms)    
        # if (max_norm-mean_norm)/(mean_norm) >= 0.5 and self.stable == 0:
        if (max_norm >1.6 or mean_norm <0.4) and self.stable == 0:
            self.stable = 1
            gradients = np.array(gradients)  # 转换为NumPy矩阵 [num_clients, grad_dim]

            similarity_matrix = cosine_similarity(gradients)
        
            clustering = AgglomerativeClustering( metric="precomputed", linkage="complete").fit(-similarity_matrix) 
            cluster_labels = clustering.labels_
            for idx, client in enumerate(clients):
                client.clu_id = int(cluster_labels[idx])

            for client in clients:
                print(f'Client ID: {client.id}, Cluster ID: {client.clu_id}')

            self.clusters = {}
            for label in set(cluster_labels):
                self.clusters[label] = [client.id for client in clients if client.clu_id == label]
            self.num_clu = len(set(cluster_labels))
            self.cluster_uploads = {cluster_id: {"ids": [], "weights": [], "models": [], "tot_samples": 0}
            for cluster_id in range(self.num_clu)
            }
            self.clusters_model = [copy.deepcopy(self.global_model) for _ in range(self.num_clu)]

        else:
            pass
    


          # 返回实际聚类数量
    def aggregate_cluster_models(self):
        """聚合各簇的模型（独立更新每个簇的全局模型）"""
        for cluster_id in self.cluster_uploads:
            upload_data = self.cluster_uploads[cluster_id]
            if len(upload_data["models"]) == 0:
                continue  # 跳过无模型的簇

            # 初始化聚合模型（深拷贝当前簇模型）
            aggregated_model = copy.deepcopy(self.clusters_model[cluster_id])

            # 加权平均聚合（FedAvg）
            for param in aggregated_model.parameters():
                param.data.zero_()

            for model, weight in zip(upload_data["models"], upload_data["weights"]):
                for agg_param, client_param in zip(aggregated_model.parameters(), model.parameters()):
                    agg_param.data += client_param.data.clone() * weight
            
            for aggre_param, server_param in zip(aggregated_model.parameters(), self.clusters_model[cluster_id].parameters()):
                server_param.data = aggre_param.data.clone()     #对服务器保存的簇的模型进行更新
        self.cluster_uploads = {cluster_id: {"ids": [], "weights": [], "models": [], "tot_samples": 0}
            for cluster_id in range(self.num_clu)   
        }  
    
    def check_cluster_stability(self):
        self.cluster_clients_by_gradients(self.clients, self.global_model)

        current_clusters = {}
        for client in self.clients:
            if client.clu_id not in current_clusters:
                current_clusters[client.clu_id] = set()
            current_clusters[client.clu_id].add(client.id)
        print(current_clusters)
        # 返回分组结构的唯一标识
        return frozenset(frozenset(cluster) for cluster in current_clusters.values())
    

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics1(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
        
        ids = [c.id for c in self.clients]
        
        return ids, num_samples, tot_correct, tot_auc
    
    #新
    def test_metrics2(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        all_preds = []
        all_labels = []
        
        for c in self.clients:
            ct, ns, auc, preds, labels = c.test_metrics()  # 修改客户端的 test_metrics 方法以返回 preds 和 labels
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            all_preds.extend(preds)
            all_labels.extend(labels)
        
        ids = [c.id for c in self.clients]
        
        return ids, num_samples, tot_correct, tot_auc, all_preds, all_labels  # 返回 preds 和 labels 
    
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        all_preds = []
        all_labels = []
        
        # 先获取所有客户端的返回值，检查返回值长度是否一致
        client_results = [c.test_metrics() for c in self.clients]
        
        # 检查所有客户端是否返回相同数量的值
        if not client_results:
            return [], [], [], [], [], []  # 没有客户端
        
        # 获取第一个客户端的返回值长度作为基准
        return_length = len(client_results[0])
        
        for res in client_results:
            if len(res) != return_length:
                raise ValueError("所有客户端的test_metrics返回值长度必须一致")
        
        ids = [c.id for c in self.clients]
        
        # 根据返回值长度决定处理方式
        if return_length == 3:
            # 处理3个返回值的情况
            for ct, ns, auc in client_results:
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)
            return ids, num_samples, tot_correct, tot_auc
        
        elif return_length == 5:
            # 处理5个返回值的情况
            for ct, ns, auc, preds, labels in client_results:
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)
                all_preds.extend(preds)
                all_labels.extend(labels)
            return ids, num_samples, tot_correct, tot_auc, all_preds, all_labels
        
        else:
            raise ValueError(f"test_metrics返回值长度{return_length}不被支持")

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate1(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
    
    #新
    def evaluate2(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        # 计算全局混淆矩阵
        cm = confusion_matrix(stats[4], stats[5])  # stats[4] 是 all_labels，stats[5] 是 all_preds
        
        # 计算每个类别的 TP, TN, FP, FN
        classes = cm.shape[0]
        tps = []
        fps = []
        fns = []
        tns = []
        for i in range(classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            tps.append(tp)
            fns.append(fn)
            fps.append(fp)
            tns.append(tn)
        
        # 计算每个类别的 TPR 和 TNR
        tprs = []
        tnrs = []
        for i in range(classes):
            if (tp + fn) > 0:
                tpr = tp / (tp + fn)
            else:
                tpr = 0
            if (tn + fp) > 0:
                tnr = tn / (tn + fp)
            else:
                tnr = 0
            tprs.append(tpr)
            tnrs.append(tnr)
        
        # 计算 G-mean
        g_mean = np.prod(tprs) ** (1 / classes)
        
        # 计算 MCC
        mcc = matthews_corrcoef(stats[4], stats[5])
        
        # 计算 F1
        f1 = f1_score(stats[4], stats[5], average='macro')
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        print("accs_list:",accs)
        
        # 打印 MCC, G-mean, F1
        print("Matthews Correlation Coefficient (MCC): {:.4f}".format(mcc))
        print("Geometric Mean (G-mean): {:.4f}".format(g_mean))
        print("F1 Score: {:.4f}".format(f1))
        # for id, c in enumerate(self.clients):
        #     print("client {} train_time_cost {}".format(stats[0][id], c.train_time_cost['total_cost']))
        #     print("client {} Test Accurancy: {:.4f}".format(stats[0][id], accs[id]))
        #     print("client {} Train Loss: {:.4f}".format(stats[0][id], stats_train[2][id] * 1.0 / stats_train[1][id]))
    
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        # 提取共用的指标计算部分
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        # 根据返回值长度决定是否计算额外的指标
        if len(stats) >= 5:
            # 如果有5个返回值，计算和打印额外的指标
            # 计算全局混淆矩阵
            cm = confusion_matrix(stats[4], stats[5])  # stats[4] 是 all_labels，stats[5] 是 all_preds

            # 计算每个类别的 TP, TN, FP, FN
            classes = cm.shape[0]
            tps = []
            fps = []
            fns = []
            tns = []
            for i in range(classes):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp
                tps.append(tp)
                fns.append(fn)
                fps.append(fp)
                tns.append(tn)

            # 计算每个类别的 TPR 和 TNR
            tprs = []
            tnrs = []
            for i in range(classes):
                if (tps[i] + fns[i]) > 0:
                    tpr = tps[i] / (tps[i] + fns[i])
                else:
                    tpr = 0
                if (tns[i] + fps[i]) > 0:
                    tnr = tns[i] / (tns[i] + fps[i])
                else:
                    tnr = 0
                tprs.append(tpr)
                tnrs.append(tnr)

            # 计算 G-mean
            g_mean = np.prod(tprs) ** (1 / classes)

            # 计算 MCC
            mcc = matthews_corrcoef(stats[4], stats[5])

            # 计算 F1
            f1 = f1_score(stats[4], stats[5], average='macro')

        # 保存和打印共用的指标
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        print("accs_list:", accs)

        # 如果有额外的指标，打印它们
        if len(stats) >= 5:
            print("Matthews Correlation Coefficient (MCC): {:.4f}".format(mcc))
            print("Geometric Mean (G-mean): {:.4f}".format(g_mean))
            print("F1 Score: {:.4f}".format(f1))



    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc




    def send_attenmodels(self,P):
        assert (len(self.clients) > 0)

        for i, client in enumerate(self.clients):
            start_time = time.time()
            
            client.set_parameters(P[i])

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
