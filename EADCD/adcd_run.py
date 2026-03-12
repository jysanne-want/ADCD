import os
import logging
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from ast import literal_eval

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ADCD_Runner:
    def __init__(self, model_class, input_dir, split_type, 
                 train_fraction, epochs, seed,
                 batch_size, lr, **model_kwargs):
        self.model_class = model_class
        self.input_dir = input_dir
        self.split_type = split_type
        self.train_fraction = train_fraction
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.model_kwargs = model_kwargs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(self.seed)

        self._load_meta_and_q_matrix()

    def _load_meta_and_q_matrix(self):
        logging.info("Loading metadata and p2k mapping...")

        metadata_path = os.path.join(self.input_dir, "meta.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.u_n = metadata['n']
        self.p_n = metadata['m']
        self.k_n = metadata['k']
        logging.info(f"{self.u_n} users, {self.p_n} problems, {self.k_n} skills.")
        
        p2k_df = pd.read_csv(os.path.join(self.input_dir, "p2k.csv"))

        logging.info("Pre-building global problem Q-matrix...")
        self.problem_q_matrix = torch.zeros((self.p_n, self.k_n))
        
        for _, row in p2k_df.iterrows():
            pid = int(row['problem_id'])
            sids = row['skill_ids']
            if isinstance(sids, str):
                sids = literal_eval(sids)
            
            if sids:
                if pid < self.p_n:
                    self.problem_q_matrix[pid, sids] = 1.0

    def _transform_to_dataloader(self, df, shuffle=True):

        u_idx = torch.tensor(df['user_id'].values, dtype=torch.int64)
        p_idx = torch.tensor(df['problem_id'].values, dtype=torch.int64)
        score = torch.tensor(df['correct'].values, dtype=torch.float32)

        q_vec = self.problem_q_matrix[p_idx]
        kc_counts_list = [json.loads(x) for x in df['kc_counts']]

        kc_counts_tensor = torch.tensor(kc_counts_list, dtype=torch.float32)
        data_set = TensorDataset(u_idx, p_idx, q_vec, kc_counts_tensor, score)
        
        return DataLoader(data_set, batch_size=self.batch_size, shuffle=shuffle)

    def _run_single_train_eval(self, train_df, val_df, test_df, fold_name=""):
        
        if self.train_fraction < 1.0:
            if self.split_type == 'real':
                n_samples = int(len(train_df) * self.train_fraction)
                train_df_reduced = train_df.iloc[:n_samples]
                logging.info(f"Real split subset: Using first {n_samples} records (ordered).")
            else:
                train_df_reduced = train_df.sample(frac=self.train_fraction, random_state=self.seed)
                logging.info(f"Random sample subset: Using {len(train_df_reduced)} records.")
        else:
            train_df_reduced = train_df

        logging.info(f"{fold_name}: Train={len(train_df_reduced)}, Val={len(val_df)}, Test={len(test_df)}")

        train_loader = self._transform_to_dataloader(train_df_reduced, shuffle=True)
        valid_loader = self._transform_to_dataloader(val_df, shuffle=False)
        test_loader = self._transform_to_dataloader(test_df, shuffle=False)

        model = self.model_class(n=self.u_n, m=self.p_n, k=self.k_n, **self.model_kwargs)

        save_dir = "paras_EADCD"
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"model_{self.split_type}_{fold_name}.snapshot")

        best_epoch = model.train(train_loader, valid_loader, self.lr, self.device, self.epochs, model_save_path)
        logging.info(f"--- {fold_name} Best epoch: {best_epoch} ---")

        logging.info(f"Loading best model from {fold_name} for final test evaluation...")
        model.load(model_save_path)
        auc, acc, rmse, f1 = model.eval(test_loader, device=self.device)
        logging.info(f"{fold_name} Test Performance --- AUC: {auc:.6f}, ACC: {acc:.6f}, RMSE: {rmse:.6f}, F1: {f1:.6f}")

        return auc, acc, rmse, f1

    def run(self):
        if self.split_type == 'random':
            self._run_random_split()
        elif self.split_type == 'weak':
            self._run_weak_split()
        elif self.split_type == 'real':
            self._run_real_split()
        else:
            raise ValueError(f"Invalid split_type: '{self.split_type}'. Choose 'random', 'weak', or 'real'.")

    def _run_random_split(self):
        fold_dir = os.path.join(self.input_dir, "folds")
        fold_results = {'auc': [], 'acc': [], 'rmse': [], 'f1': []}

        for fold in range(1, 6):
            logging.info(f"\n{'=' * 20} Starting Random Split Fold {fold}/5 {'=' * 20}")
            fold_name = f"random_fold_{fold}"

            train_val_df = pd.read_csv(os.path.join(fold_dir, f'tv_fold_{fold}.csv'))
            test_df = pd.read_csv(os.path.join(fold_dir, f'test_fold_{fold}.csv'))

            train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=self.seed)

            auc, acc, rmse, f1 = self._run_single_train_eval(train_df, val_df, test_df, fold_name)

            fold_results['auc'].append(auc)
            fold_results['acc'].append(acc)
            fold_results['rmse'].append(rmse)
            fold_results['f1'].append(f1)

        self._print_summary(fold_results, title="5-Fold Cross-Validation Summary (Random Split)")

    def _run_weak_split(self):
        logging.info(f"\n{'=' * 20} Starting Weak-Coverage Split {'=' * 20}")
        weak_dir = os.path.join(self.input_dir, "weak")
        fold_name = "weak_split"

        train_val_df = pd.read_csv(os.path.join(weak_dir, 'tv.csv'))
        test_df = pd.read_csv(os.path.join(weak_dir, 'test.csv'))

        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=self.seed)

        auc, acc, rmse, f1 = self._run_single_train_eval(train_df, val_df, test_df, fold_name)

        print(f"\n{'=' * 20} Weak-Coverage Split Final Results {'=' * 20}")
        print(f"Parameters: train_fraction={self.train_fraction}, epochs={self.epochs}, seed={self.seed}")
        print(f"AUC:  {auc:.6f}")
        print(f"ACC:  {acc:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"F1:   {f1:.6f}")

    def _run_real_split(self):
        logging.info(f"\n{'=' * 20} Starting Real-Scenario (Time-Series) Split {'=' * 20}")
        real_dir = os.path.join(self.input_dir, "real")
        fold_name = "real_split"

        train_df = pd.read_csv(os.path.join(real_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(real_dir, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(real_dir, 'test.csv'))
        

        auc, acc, rmse, f1 = self._run_single_train_eval(train_df, val_df, test_df, fold_name)

        print(f"\n{'=' * 20} Real-Scenario (Time-Series) Split Final Results {'=' * 20}")
        print(f"Parameters: train_fraction={self.train_fraction}, epochs={self.epochs}, seed={self.seed}")
        print(f"AUC:  {auc:.6f}")
        print(f"ACC:  {acc:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"F1:   {f1:.6f}")

    def _print_summary(self, results, title):
        print(f"\n{'=' * 20} {title} {'=' * 20}")
        print(f"Parameters: train_fraction={self.train_fraction}, epochs={self.epochs}, seed={self.seed}")
        print(f"Average AUC:  {np.mean(results['auc']):.6f} ± {np.std(results['auc']):.6f}")
        print(f"Average ACC:  {np.mean(results['acc']):.6f} ± {np.std(results['acc']):.6f}")
        print(f"Average RMSE: {np.mean(results['rmse']):.6f} ± {np.std(results['rmse']):.6f}")
        print(f"Average F1:   {np.mean(results['f1']):.6f} ± {np.std(results['f1']):.6f}")