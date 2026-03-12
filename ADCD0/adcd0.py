from EduCDM import CDM
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class ADCD0_Net(nn.Module):
    def __init__(self, n, m, k, dim):
        super(ADCD0_Net, self).__init__()
        self.n = n
        self.m = m
        self.k = k
        self.dim = dim
        
        self.theta = nn.Embedding(self.n, 1) 
        self.delta_emb = nn.Embedding(self.n, self.dim) 

        self.v_k = nn.Linear(self.k, self.dim) 
        self.fusion_gate = nn.Linear(self.dim * 2, self.dim)
        
        self.gamma = nn.Embedding(self.k, self.dim)
        self.dk_projector = nn.Linear(self.dim, 1)
        self.exer = nn.Embedding(self.m, self.dim)
        self.disc = nn.Embedding(self.m, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.prednet_input_len = self.k
        self.prednet_len1, self.prednet_len2 = 256, 128
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, s_id, e_id, q_vec, kc_counts):
        theta_s = self.theta(s_id)
        
        u_base = self.delta_emb(s_id) 

        log_kc_counts_vec = torch.log(1.0 + kc_counts.float())
        h_exp = torch.tanh(self.v_k(log_kc_counts_vec))
        
        combined = torch.cat([u_base, h_exp], dim=1)
        g = torch.sigmoid(self.fusion_gate(combined))
        u_final = (1 - g) * u_base + g * h_exp

        gamma_k = self.gamma.weight
        dk = torch.sigmoid(self.dk_projector(gamma_k)).squeeze(-1)
        
        lf = self.alpha * theta_s - dk * torch.exp(-self.beta * theta_s)
        delta_sk = torch.matmul(u_final, gamma_k.T)
        p_sk = torch.sigmoid(lf + delta_sk)

        e = self.exer(e_id)
        d_ek = torch.sigmoid(torch.matmul(e, gamma_k.T))
        disc_e = torch.sigmoid(self.disc(e_id))

        input_x = q_vec * (p_sk - d_ek) * disc_e
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output.view(-1)


class ADCD0(CDM):
    def __init__(self, n, m, k, dim):
        super(ADCD0, self).__init__()
        self.net = ADCD0_Net(n, m, k, dim)

    def train(self, train, valid, lr, device, epoch, save_path):
        self.net = self.net.to(device)
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        best_auc = 0.0
        threshold = 0.001
        
        for i in range(epoch):
            self.net.train()
            epoch_loss = []
            
            for batch in tqdm(train, f"Epoch {i + 1}/{epoch}"):
                s_id, e_id, k_vec, kc_counts, y = batch 
                s_id, e_id, k_vec, kc_counts, y = \
                    s_id.to(device), e_id.to(device), k_vec.to(device), kc_counts.to(device), y.to(device)
                
                pred = self.net(s_id, e_id, k_vec, kc_counts)
                loss = loss_func(pred, y.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

                with torch.no_grad():
                    self.net.alpha.data.clamp_(min=1e-8)
                    self.net.beta.data.clamp_(min=1e-8)
            
            if valid is not None:
                auc, acc, rmse, f1 = self.eval(valid, device)
                print(f"\n[Epoch {i + 1}] Validation AUC: {auc:.6f}")

                if auc > best_auc + threshold:
                    best_auc = auc
                    torch.save(self.net.state_dict(), save_path)
                    print(f"  [*] Valid Improvement. Model saved. Best AUC: {best_auc:.6f}")
                else:
                    print(f"  [!] No improvement. Stopping.")
                    break 

        return best_auc
    
    def eval(self, test, device):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in tqdm(test, "Evaluating"):
                s_id, e_id, k_vec, kc_counts, y = batch 
                s_id, e_id, k_vec, kc_counts = \
                    s_id.to(device), e_id.to(device), k_vec.to(device), kc_counts.to(device)
                pred = self.net(s_id, e_id, k_vec, kc_counts)
                y_true.extend(y.numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
               np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))