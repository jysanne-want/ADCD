from torch.utils.data import Dataset, TensorDataset, DataLoader
from collections import Counter
from adcd_run import ADCD_Runner
from sklearn.cluster import AgglomerativeClustering
import os
import json
import torch
import random
import logging
import numpy as np

class CMESTrainDataset(Dataset):
    def __init__(self, df, problem_q_matrix, clusters, user_interactions, topk, p_n):
        self.u_idx = df['user_id'].values
        self.p_idx = df['problem_id'].values
        self.score = df['correct'].values
        self.problem_q_matrix = problem_q_matrix
        self.topk = topk
        self.p_n = p_n

        # --- 核心优化：预处理采样池 ---
        logging.info("Pre-calculating sampling pools for each user... This may take a minute.")
        self.user_sampling_pools = {}
        
        # 1. 按簇分组用户
        cluster_to_users = {}
        for u, c in clusters.items():
            cluster_to_users.setdefault(c, []).append(u)
        
        # 2. 为每个用户预计算候选池 (Peer 做过但我没做过的题)
        all_uids = df['user_id'].unique()
        for uid in all_uids:
            cur_cluster = clusters.get(uid, -1)
            cur_user_history = user_interactions.get(uid, set())
            
            # 找到其他簇的用户
            other_clusters = [c for c in cluster_to_users.keys() if c != cur_cluster]
            
            # 这里的逻辑：从其他簇中选出最热门的题目，作为该用户的候选池
            pool_candidates = []
            for c in other_clusters:
                # 为了性能，每个簇只选一部分用户代表
                peers = cluster_to_users[c]
                sampled_peers = random.sample(peers, min(len(peers), 10))
                for peer in sampled_peers:
                    pool_candidates.extend(list(user_interactions.get(peer, set())))
            
            # 过滤掉自己做过的
            final_pool = list(set(pool_candidates) - cur_user_history)
            
            # 如果池子太小，用全集补充一点，防止后面报错
            if len(final_pool) < topk:
                final_pool.extend(random.sample(range(p_n), topk))
            
            self.user_sampling_pools[uid] = np.array(final_pool)
            
        logging.info("Sampling pools ready.")

    def __len__(self):
        return len(self.u_idx)

    def __getitem__(self, idx):
        uid = self.u_idx[idx]
        pid = self.p_idx[idx]
        y = self.score[idx]
        q_vec = self.problem_q_matrix[pid]

        # --- 极速采样：直接从预设好的池子里抽样 ---
        pool = self.user_sampling_pools.get(uid)
        
        # 随机抽取 topk 个
        if len(pool) >= self.topk:
            sampled_ids = np.random.choice(pool, self.topk, replace=False)
        else:
            sampled_ids = np.random.choice(pool, self.topk, replace=True)

        return torch.tensor(uid, dtype=torch.long), \
               torch.tensor(pid, dtype=torch.long), \
               q_vec, \
               torch.tensor(y, dtype=torch.float32), \
               torch.tensor(sampled_ids, dtype=torch.long)

class CMES_Runner(ADCD_Runner):
    def __init__(self, model_class, input_dir, split_type, 
                 train_fraction, epochs, seed,
                 batch_size, lr, 
                 n_clusters=20, topk=20, # [新增] CMES 超参数
                 **model_kwargs):
        
        super().__init__(model_class, input_dir, split_type, 
                         train_fraction, epochs, seed,
                         batch_size, lr, **model_kwargs)
        self.n_clusters = n_clusters
        self.topk = topk
        self.clusters = None # 存储聚类结果
        self.user_interactions = None # 存储交互历史

    def _perform_clustering(self, full_df):
        logging.info(f"Performing User Clustering (n_clusters={self.n_clusters})...")
        
        # --- 优化点：预计算题目到知识点的映射，避免在循环中调用 torch.nonzero ---
        # 得到一个列表，索引是 pid，值是该题目包含的知识点索引列表
        problem_to_ks = [torch.nonzero(self.problem_q_matrix[i]).flatten().numpy() 
                        for i in range(self.p_n)]
        
        user_k_matrix = np.zeros((self.u_n, self.k_n))
        user_k_matrix.fill(-1)
        
        # 使用 values 加速读取
        uids = full_df['user_id'].values
        pids = full_df['problem_id'].values
        corrects = full_df['correct'].values

        for i in range(len(uids)):
            uid = uids[i]
            if uid >= self.u_n: continue
            pid = pids[i]
            score = corrects[i]
            
            # 直接从预计算的列表中取，极快
            for k in problem_to_ks[pid]:
                user_k_matrix[uid, k] = score

        # 2. 执行聚类 (Agglomerative 在 4万数据上可能稍慢，但能跑)
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters)
        labels = clustering.fit_predict(user_k_matrix)
        
        # 3. 保存结果
        self.clusters = {uid: label for uid, label in enumerate(labels)}
        logging.info("Clustering completed.")

    # [新增] 构建交互历史索引
    def _build_interaction_index(self, df):
        logging.info("Building user interaction index...")
        self.user_interactions = {}
        grouped = df.groupby('user_id')['problem_id'].apply(set)
        for uid, p_set in grouped.items():
            self.user_interactions[uid] = p_set

    # [重写] 核心改动：根据 mode 选择不同的 DataLoader
    def _transform_to_dataloader(self, df, shuffle=True, is_train=False):
        
        if is_train:
            # 训练集：使用 CMES 自定义 Dataset
            dataset = CMESTrainDataset(
                df=df,
                problem_q_matrix=self.problem_q_matrix,
                clusters=self.clusters,
                user_interactions=self.user_interactions,
                topk=self.topk,
                p_n=self.p_n
            )
            return DataLoader(dataset, batch_size=self.batch_size, 
                          shuffle=shuffle, num_workers=4, pin_memory=True)
        else:
            # 验证/测试集：保持原有的 TensorDataset (不需要采样)
            u_idx = torch.tensor(df['user_id'].values, dtype=torch.int64)
            p_idx = torch.tensor(df['problem_id'].values, dtype=torch.int64)
            score = torch.tensor(df['correct'].values, dtype=torch.float32)
            q_vec = self.problem_q_matrix[p_idx]
            
            # 注意：原有逻辑中你处理了 kc_counts，如果 CMES 模型不需要这个，可以去掉
            # 如果需要兼容，请保持
            if 'kc_counts' in df.columns:
                 kc_counts_list = [json.loads(x) for x in df['kc_counts']]
                 kc_counts_tensor = torch.tensor(kc_counts_list, dtype=torch.float32)
                 data_set = TensorDataset(u_idx, p_idx, q_vec, kc_counts_tensor, score)
            else:
                 data_set = TensorDataset(u_idx, p_idx, q_vec, score)

            return DataLoader(data_set, batch_size=self.batch_size, 
                        shuffle=shuffle, num_workers=4, pin_memory=True)

    # [重写] 单次训练评估流程
    def _run_single_train_eval(self, train_df, val_df, test_df, fold_name=""):
        
        # 1. 在任何数据切分之前或之后，必须先进行聚类和构建索引
        # 通常基于训练集的数据构建，防止数据泄露
        self._build_interaction_index(train_df)
        self._perform_clustering(train_df) # 仅用训练集聚类
        
        if self.train_fraction < 1.0:
            train_df_reduced = train_df.sample(frac=self.train_fraction, random_state=self.seed)
        else:
            train_df_reduced = train_df

        # 2. 创建 DataLoader (注意 is_train 标志)
        train_loader = self._transform_to_dataloader(train_df_reduced, shuffle=True, is_train=True)
        valid_loader = self._transform_to_dataloader(val_df, shuffle=False, is_train=False)
        test_loader = self._transform_to_dataloader(test_df, shuffle=False, is_train=False)

        # 3. 初始化模型 (传入 topk)
        model = self.model_class(n=self.u_n, m=self.p_n, k=self.k_n, topk=self.topk, **self.model_kwargs)

        save_dir = "paras_CMES"
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"model_{self.split_type}_{fold_name}.snapshot")

        # 4. 训练
        best_epoch = model.train(train_loader, valid_loader, self.lr, self.device, self.epochs, model_save_path)
        
        # 5. 评估
        model.load(model_save_path)
        auc, acc, rmse, f1 = model.eval(test_loader, device=self.device)
        logging.info(f"{fold_name} Test Results: AUC={auc:.4f}")

        return auc, acc, rmse, f1
    
    def run_single_fold(self, fold=1):
        """
        运行单次交叉验证折叠，并返回验证集AUC。
        专为 Optuna 调参设计，默认使用随机切分的第一折数据。
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        logging.info(f"\n{'=' * 20} Starting Optuna Trial on Fold {fold} {'=' * 20}")
        
        # 1. 确定数据路径
        # 假设数据存放在 input_dir/folds 目录下，格式为 tv_fold_1.csv 和 test_fold_1.csv
        fold_dir = os.path.join(self.input_dir, "folds")
        fold_name = f"optuna_trial_fold_{fold}"

        try:
            # 读取训练验证集 (Train+Val) 和 测试集 (Test)
            # 注意：Optuna 调参主要关注验证集表现，所以这里 Test 集其实可以不读，但为了复用流程还是保留
            train_val_df = pd.read_csv(os.path.join(fold_dir, f'tv_fold_{fold}.csv'))
            test_df = pd.read_csv(os.path.join(fold_dir, f'test_fold_{fold}.csv'))
        except FileNotFoundError:
            # 如果找不到 fold 文件，尝试回退到 weak 或 real 目录 (兼容性处理)
            if self.split_type == 'weak':
                 weak_dir = os.path.join(self.input_dir, "weak")
                 train_val_df = pd.read_csv(os.path.join(weak_dir, 'tv.csv'))
                 test_df = pd.read_csv(os.path.join(weak_dir, 'test.csv'))
            elif self.split_type == 'real':
                 real_dir = os.path.join(self.input_dir, "real")
                 # Real split 通常已经是 train/valid/test 分好的
                 train_df = pd.read_csv(os.path.join(real_dir, 'train.csv'))
                 val_df = pd.read_csv(os.path.join(real_dir, 'valid.csv'))
                 test_df = pd.read_csv(os.path.join(real_dir, 'test.csv'))
                 # 提前返回，跳过下面的切分
                 return self._run_optuna_eval(train_df, val_df, test_df, fold_name)
            else:
                logging.error(f"Fold {fold} data not found in {fold_dir}. Please check your data split.")
                return 0.0

        # 2. 切分训练集和验证集 (Train / Val)
        # 即使是 weak 模式，也需要切分出验证集来给 Optuna 看
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=self.seed)

        # 3. 执行训练和评估
        # 我们创建一个辅助方法 _run_optuna_eval 来复用逻辑并捕获验证集 AUC
        return self._run_optuna_eval(train_df, val_df, test_df, fold_name)

    def _run_optuna_eval(self, train_df, val_df, test_df, fold_name):
        """
        run_single_fold 的辅助方法，执行实际的训练并返回 Valid AUC
        """
        # A. 聚类和索引构建 (核心前置步骤)
        self._build_interaction_index(train_df)
        self._perform_clustering(train_df)
        
        # B. 数据降采样 (如果 train_fraction < 1.0)
        if self.train_fraction < 1.0:
            train_df_reduced = train_df.sample(frac=self.train_fraction, random_state=self.seed)
        else:
            train_df_reduced = train_df

        # C. 构建 DataLoader
        train_loader = self._transform_to_dataloader(train_df_reduced, shuffle=True, is_train=True)
        valid_loader = self._transform_to_dataloader(val_df, shuffle=False, is_train=False)
        # Optuna 阶段其实不需要 Test Loader，为了节省时间可以传 None，但需要 wrapper 支持
        # 这里为了稳健还是传进去
        test_loader = self._transform_to_dataloader(test_df, shuffle=False, is_train=False)

        # D. 初始化模型
        model = self.model_class(n=self.u_n, m=self.p_n, k=self.k_n, topk=self.topk, **self.model_kwargs)

        save_dir = "paras_CMES_optuna"
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"model_{fold_name}.snapshot")

        # E. 训练
        # 注意：这里我们只关心验证集 AUC，不需要测试集结果
        model.train(train_loader, valid_loader, self.lr, self.device, self.epochs, model_save_path)
        
        # F. 加载最佳模型并在验证集上再次评估以获取最终 AUC
        model.load(model_save_path)
        val_auc, _, _, _ = model.eval(valid_loader, device=self.device)
        
        logging.info(f"[Optuna] Trial Finished. Validation AUC: {val_auc:.6f}")
        return val_auc