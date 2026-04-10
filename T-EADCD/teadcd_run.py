import os
import logging
import json
from ast import literal_eval

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

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

    def _validate_required_columns(self, df, frame_name):
        required_cols = {"user_id", "problem_id", "correct", "kc_counts"}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            missing_str = ", ".join(sorted(missing_cols))
            raise ValueError(f"{frame_name} is missing required columns: {missing_str}")

        validated_df = df.copy()
        validated_df["correct"] = pd.to_numeric(validated_df["correct"], errors="coerce")
        validated_df.dropna(subset=["user_id", "problem_id", "correct", "kc_counts"], inplace=True)

        if "rt" in validated_df.columns:
            validated_df["rt"] = pd.to_numeric(validated_df["rt"], errors="coerce").fillna(0.0)
        else:
            logging.warning("%s does not contain an explicit rt column. Falling back to kc_counts parsing.", frame_name)

        return validated_df

    def _parse_sequence_field(self, value):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return literal_eval(stripped)
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        if pd.isna(value):
            return []
        return value

    def _split_kc_counts_and_rt(self, value, explicit_rt=None):
        parsed = self._parse_sequence_field(value)
        rt_value = None if explicit_rt is None or pd.isna(explicit_rt) else float(explicit_rt)

        if isinstance(parsed, dict):
            counts = np.zeros(self.k_n, dtype=np.float32)
            for key, count in parsed.items():
                idx = int(key)
                if 0 <= idx < self.k_n:
                    counts[idx] = float(count)
            return counts, 0.0 if rt_value is None else rt_value

        if isinstance(parsed, (list, tuple, np.ndarray)):
            values = np.asarray(parsed, dtype=np.float32).reshape(-1)
            if rt_value is None and values.shape[0] == self.k_n + 1:
                rt_value = float(values[0])
                values = values[1:]

            counts = values
            if counts.shape[0] == self.k_n:
                return counts, 0.0 if rt_value is None else rt_value
            if counts.shape[0] > self.k_n:
                return counts[:self.k_n], 0.0 if rt_value is None else rt_value

            padded = np.zeros(self.k_n, dtype=np.float32)
            padded[:counts.shape[0]] = counts
            return padded, 0.0 if rt_value is None else rt_value

        raise ValueError(f"Unsupported kc_counts format: {type(value)}")

    def _ensure_rt_and_counts(self, df):
        df = df.copy()
        explicit_rt_values = df["rt"].values if "rt" in df.columns else [None] * len(df)
        counts_list, rt_list = [], []

        for kc_value, rt_value in zip(df["kc_counts"], explicit_rt_values):
            counts, resolved_rt = self._split_kc_counts_and_rt(kc_value, rt_value)
            counts_list.append(counts)
            rt_list.append(float(resolved_rt))

        df["kc_counts_parsed"] = counts_list
        df["rt"] = np.asarray(rt_list, dtype=np.float32)
        return df

    def _normalize_feature(self, values, mean, std):
        normalized = (values - mean) / std
        return np.clip(normalized, -5.0, 5.0)

    def _attach_time_context(self, train_df, target_df):
        target_df = target_df.copy()

        train_rt_raw = np.log1p(train_df["rt"].clip(lower=0).to_numpy(dtype=np.float32))
        rt_mean = float(train_rt_raw.mean())
        rt_std = float(train_rt_raw.std())
        rt_std = rt_std if rt_std > 1e-6 else 1.0

        user_mean_rt = train_df.groupby("user_id")["rt"].mean()
        item_mean_rt = train_df.groupby("problem_id")["rt"].mean()
        user_speed_raw_map = -np.log1p(user_mean_rt)
        item_load_raw_map = np.log1p(item_mean_rt)

        train_user_speed_raw = train_df["user_id"].map(user_speed_raw_map).to_numpy(dtype=np.float32)
        train_item_load_raw = train_df["problem_id"].map(item_load_raw_map).to_numpy(dtype=np.float32)

        speed_mean = float(train_user_speed_raw.mean())
        speed_std = float(train_user_speed_raw.std())
        speed_std = speed_std if speed_std > 1e-6 else 1.0

        load_mean = float(train_item_load_raw.mean())
        load_std = float(train_item_load_raw.std())
        load_std = load_std if load_std > 1e-6 else 1.0

        default_speed_raw = float(user_speed_raw_map.mean())
        default_load_raw = float(item_load_raw_map.mean())

        target_rt_raw = np.log1p(target_df["rt"].clip(lower=0).to_numpy(dtype=np.float32))
        target_df["rt_norm"] = self._normalize_feature(target_rt_raw, rt_mean, rt_std)

        target_user_speed_raw = (
            target_df["user_id"].map(user_speed_raw_map).fillna(default_speed_raw).to_numpy(dtype=np.float32)
        )
        target_item_load_raw = (
            target_df["problem_id"].map(item_load_raw_map).fillna(default_load_raw).to_numpy(dtype=np.float32)
        )

        target_df["user_speed"] = self._normalize_feature(target_user_speed_raw, speed_mean, speed_std)
        target_df["item_load"] = self._normalize_feature(target_item_load_raw, load_mean, load_std)
        return target_df

    def _prepare_split_features(self, train_df, val_df, test_df):
        train_df = self._validate_required_columns(train_df, "train_df")
        val_df = self._validate_required_columns(val_df, "val_df")
        test_df = self._validate_required_columns(test_df, "test_df")

        train_df = self._ensure_rt_and_counts(train_df)
        val_df = self._ensure_rt_and_counts(val_df)
        test_df = self._ensure_rt_and_counts(test_df)

        train_df = self._attach_time_context(train_df, train_df)
        val_df = self._attach_time_context(train_df, val_df)
        test_df = self._attach_time_context(train_df, test_df)
        return train_df, val_df, test_df

    def _load_meta_and_q_matrix(self):
        logging.info("Loading metadata and p2k mapping...")

        metadata_path = os.path.join(self.input_dir, "meta.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.u_n, self.p_n, self.k_n = metadata['n'], metadata['m'], metadata['k']
        logging.info(f"{self.u_n} users, {self.p_n} problems, {self.k_n} skills.")

        p2k_df = pd.read_csv(os.path.join(self.input_dir, "p2k.csv"))
        logging.info("Pre-building global problem Q-matrix...")
        self.problem_q_matrix = torch.zeros((self.p_n, self.k_n), dtype=torch.float32)

        for _, row in p2k_df.iterrows():
            pid = int(row['problem_id'])
            skill_ids = self._parse_sequence_field(row['skill_ids'])

            if not skill_ids or pid < 0 or pid >= self.p_n:
                continue

            valid_sids = []
            for sid in skill_ids:
                sid = int(sid)
                if 0 <= sid < self.k_n:
                    valid_sids.append(sid)

            if valid_sids:
                self.problem_q_matrix[pid, valid_sids] = 1.0

    def _transform_to_dataloader(self, df, shuffle=True):
        u_idx = torch.tensor(df['user_id'].values, dtype=torch.int64)
        p_idx = torch.tensor(df['problem_id'].values, dtype=torch.int64)
        score = torch.tensor(df['correct'].values, dtype=torch.float32)

        q_vec = self.problem_q_matrix[p_idx]
        kc_counts_array = np.stack(df['kc_counts_parsed'].to_numpy())
        kc_counts_tensor = torch.tensor(kc_counts_array, dtype=torch.float32)
        rt_tensor = torch.tensor(df['rt_norm'].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
        user_speed_tensor = torch.tensor(df['user_speed'].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
        item_load_tensor = torch.tensor(df['item_load'].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32)

        data_set = TensorDataset(
            u_idx,
            p_idx,
            q_vec,
            kc_counts_tensor,
            rt_tensor,
            user_speed_tensor,
            item_load_tensor,
            score,
        )
        return DataLoader(data_set, batch_size=self.batch_size, shuffle=shuffle)

    def _run_single_train_eval(self, train_df, val_df, test_df, fold_name=""):
        if self.train_fraction < 1.0:
            if self.split_type == 'real':
                n_samples = int(len(train_df) * self.train_fraction)
                train_df_reduced = train_df.iloc[:n_samples]
                logging.info("Real split subset: Using first %s records (ordered).", n_samples)
            else:
                train_df_reduced = train_df.sample(frac=self.train_fraction, random_state=self.seed)
                logging.info("Random sample subset: Using %s records.", len(train_df_reduced))
        else:
            train_df_reduced = train_df

        logging.info("%s: Train=%s, Val=%s, Test=%s", fold_name, len(train_df_reduced), len(val_df), len(test_df))

        train_df_ready, val_df_ready, test_df_ready = self._prepare_split_features(
            train_df_reduced, val_df, test_df
        )

        train_loader = self._transform_to_dataloader(train_df_ready, shuffle=True)
        valid_loader = self._transform_to_dataloader(val_df_ready, shuffle=False)
        test_loader = self._transform_to_dataloader(test_df_ready, shuffle=False)

        model = self.model_class(n=self.u_n, m=self.p_n, k=self.k_n, **self.model_kwargs)

        save_dir = "paras_EADCD"
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"model_{self.split_type}_{fold_name}.snapshot")

        best_epoch = model.train(train_loader, valid_loader, self.lr, self.device, self.epochs, model_save_path)
        logging.info("--- %s Best epoch: %s ---", fold_name, best_epoch)

        logging.info("Loading best model from %s for final test evaluation...", fold_name)
        model.load(model_save_path)
        auc, acc, rmse, f1 = model.eval(test_loader, device=self.device)
        logging.info(
            "%s Test Performance --- AUC: %.6f, ACC: %.6f, RMSE: %.6f, F1: %.6f",
            fold_name,
            auc,
            acc,
            rmse,
            f1,
        )

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
            logging.info("\n%s Starting Random Split Fold %s/5 %s", '=' * 20, fold, '=' * 20)
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
        logging.info("\n%s Starting Weak-Coverage Split %s", '=' * 20, '=' * 20)
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
        logging.info("\n%s Starting Real-Scenario (Time-Series) Split %s", '=' * 20, '=' * 20)
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
        print(f"Average AUC:  {np.mean(results['auc']):.6f} +/- {np.std(results['auc']):.6f}")
        print(f"Average ACC:  {np.mean(results['acc']):.6f} +/- {np.std(results['acc']):.6f}")
        print(f"Average RMSE: {np.mean(results['rmse']):.6f} +/- {np.std(results['rmse']):.6f}")
        print(f"Average F1:   {np.mean(results['f1']):.6f} +/- {np.std(results['f1']):.6f}")
