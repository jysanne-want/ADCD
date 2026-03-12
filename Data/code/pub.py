import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

def calc_weak_r(train_df, test_df, p2k):

    if len(test_df) == 0: 
        return 0.0
    
    tr_temp = train_df[['user_id', 'problem_id']].copy()
    tr_temp['sks'] = tr_temp['problem_id'].map(p2k)
    user_known = tr_temp.explode('sks').groupby('user_id')['sks'].apply(set).to_dict()
    
    te_temp = test_df[['user_id', 'problem_id']].copy()
    te_temp['sks'] = te_temp['problem_id'].map(p2k)
    
    weak_count = 0
    total_count = len(te_temp)
    
    for uid, q_kcs in zip(te_temp['user_id'], te_temp['sks']):
        known = user_known.get(uid, set())
        
        if isinstance(q_kcs, list) and len(q_kcs) > 0:
            unknown_num = sum(1 for k in q_kcs if k not in known)
            
            if (unknown_num / len(q_kcs)) > 0.5:
                weak_count += 1
                
    return weak_count / total_count

class Weak:
    def __init__(self, op_dir, seed=123):
        self.op_dir = os.path.join(op_dir, 'weak')
        self.seed = seed
        self.cols = ['user_id', 'problem_id', 'correct', 'kc_counts']
        os.makedirs(self.op_dir, exist_ok=True)

    def split_and_save(self, df, p2k):
        print("\n" + "="*10 + " 1 Weak-Split (Corrected) " + "="*10)
        tv_df, test_df = [], []
        temp_df = df.copy()
        temp_df['skill_ids'] = temp_df['problem_id'].map(p2k)

        temp_df = temp_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        for user_id, group in tqdm(temp_df.groupby('user_id'), "Weak Split"):
            sks = [k for ks in group['skill_ids'].dropna() for k in ks]
            if not sks:
                n_test = int(0.2 * len(group))
                test_df.append(group.iloc[:n_test])
                tv_df.append(group.iloc[n_test:])
                continue
                
            k_counts = pd.Series(sks).value_counts().to_dict()

            def get_sparsity_score(skill_list):
                if not skill_list: return 0
                return sum(1.0 / k_counts.get(k, 99999) for k in skill_list)

            group = group.copy()
            group['score'] = group['skill_ids'].apply(get_sparsity_score)
            group.sort_values(by='score', ascending=False, inplace=True)
            
            n_test = int(0.2 * len(group))
            s_test_df = group.iloc[:n_test]
            s_tv_df = group.iloc[n_test:]

            tv_df.append(s_tv_df)
            test_df.append(s_test_df)

        final_tv_df = pd.concat(tv_df, ignore_index=True)[self.cols]
        final_test_df = pd.concat(test_df, ignore_index=True)[self.cols]

        w_ratio = calc_weak_r(final_tv_df, final_test_df, p2k)
        print(f"Weak response proportion: {w_ratio:.3f}")

        final_tv_df.to_csv(os.path.join(self.op_dir, 'tv.csv'), index=False)
        final_test_df.to_csv(os.path.join(self.op_dir, 'test.csv'), index=False)
        
        print(f"Weak Split Done. tv: {len(final_tv_df)}, test: {len(final_test_df)}")

class Random:
    def __init__(self, op_dir, n_folds=5, seed=123):
        self.op_dir = os.path.join(op_dir, 'folds')
        self.n_folds = n_folds
        self.seed = seed
        self.cols = ['user_id', 'problem_id', 'correct', 'kc_counts']
        os.makedirs(self.op_dir, exist_ok=True)
        
    def split_and_save(self, df, p2k):
        print("\n" + "="*10 + " 2 Random 5-Fold Split " + "="*10)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        folds_data = [[] for _ in range(self.n_folds)]
        for user_id, group in tqdm(df.groupby('user_id'), "Random Split"):
            if len(group) < self.n_folds:
                indices = np.arange(len(group))
                np.random.shuffle(indices)
                for i in range(self.n_folds):
                    folds_data[i].append(group.iloc[indices[i::self.n_folds]])
            else:
                for i, (_, fold_indices) in enumerate(kf.split(group)):
                    folds_data[i].append(group.iloc[fold_indices])

        weak_rs = []

        for i in range(self.n_folds):
            tv_df_list = [pd.concat(folds_data[j]) for j in range(self.n_folds) if i != j]
            tv_df = pd.concat(tv_df_list, ignore_index=True)
            test_df = pd.concat(folds_data[i], ignore_index=True)
            
            r = calc_weak_r(tv_df, test_df, p2k)
            weak_rs.append(r)
            
            tv_df[self.cols].to_csv(os.path.join(self.op_dir, f'tv_fold_{i + 1}.csv'), index=False)
            test_df[self.cols].to_csv(os.path.join(self.op_dir, f'test_fold_{i + 1}.csv'), index=False)
        
        print(f"Random 5-Fold Split Done. Avg Weak response proportion: {np.mean(weak_rs):.3f}")


class Real:
    def __init__(self, op_dir, train_r=0.6, valid_r=0.2):
        self.op_dir = os.path.join(op_dir, 'real')
        self.train_r = train_r
        self.valid_r = valid_r
        self.cols = ['user_id', 'problem_id', 'correct', 'kc_counts']
        os.makedirs(self.op_dir, exist_ok=True)
        
    def split_and_save(self, df, p2k): 
        print("\n" + "="*10 + " 3 Real-Scenario (Time-Series) Split " + "="*10)
        
        train_dfs, valid_dfs, test_dfs = [], [], []
        for user_id, group in tqdm(df.groupby('user_id'), "Time-Series Split"):
            n = len(group)
            n_train = int(n * self.train_r)
            n_valid = int(n * self.valid_r)
            
            train_dfs.append(group.iloc[:n_train])
            valid_dfs.append(group.iloc[n_train : n_train + n_valid])
            test_dfs.append(group.iloc[n_train + n_valid:])

        final_train = pd.concat(train_dfs, ignore_index=True)
        final_valid = pd.concat(valid_dfs, ignore_index=True)
        final_test = pd.concat(test_dfs, ignore_index=True)

        combined_train = pd.concat([final_train, final_valid], ignore_index=True)
        w_ratio = calc_weak_r(combined_train, final_test, p2k)
        print(f"Weak response proportion: {w_ratio:.3f}")

        final_train[self.cols].to_csv(os.path.join(self.op_dir, 'train.csv'), index=False)
        final_valid[self.cols].to_csv(os.path.join(self.op_dir, 'valid.csv'), index=False)
        final_test[self.cols].to_csv(os.path.join(self.op_dir, 'test.csv'), index=False)

        print(f"Real Split Done. Train: {len(final_train)}, Val: {len(final_valid)}, Test: {len(final_test)}")
