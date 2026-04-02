import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class EADCD_Net(nn.Module):
    def __init__(self, n, m, k, dim, emb_dropout=0.3):
        super(EADCD_Net, self).__init__()
        self.n = n
        self.m = m
        self.k = k
        self.dim = dim
        self.emb_dropout = emb_dropout

        self.theta = nn.Embedding(self.n, 1)        # Gf
        self.delta = nn.Embedding(self.n, self.dim) # Gc

        self.generator = nn.Sequential(nn.Linear(1, self.dim // 2), nn.Tanh(), nn.Linear(self.dim // 2, self.dim))
        self.extractor = nn.Sequential(nn.Linear(self.dim, self.dim // 2), nn.Tanh(), nn.Linear(self.dim // 2, 1))
        self.belief_gate = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

        self.gamma = nn.Embedding(self.k, self.dim) # vk
        
        self.local_growth_mod = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1), nn.Softplus())
        self.interference_mlp = nn.Sequential(nn.Linear(self.dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.transfer_mod = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid())

        self.experience_encoder = nn.Linear(self.k, self.dim) 
        self.gain_gate = nn.Linear(self.dim * 2, self.dim)
        self.exer = nn.Embedding(self.m, self.dim)
        self.disc = nn.Embedding(self.m, 1)
        self.dk_projector = nn.Linear(self.dim, 1)
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
            if 'weight' in name: nn.init.xavier_normal_(param)

    def forward(self, s_id, e_id, q_vec, kc_counts):

        theta_raw = self.theta(s_id)
        u_raw = self.delta(s_id)
        total_counts = kc_counts.sum(dim=1, keepdim=True)
        log_total = torch.log(1.0 + total_counts)

        u_prior = self.generator(theta_raw)
        alpha = self.belief_gate(log_total)
        
        if self.training and self.emb_dropout > 0:
            mask = (torch.rand(s_id.shape[0], 1, device=s_id.device) > self.emb_dropout).float()
            alpha_final = alpha * mask
        else:
            alpha_final = alpha
            
        u_base = (1 - alpha_final) * u_prior + alpha_final * u_raw

        theta_pred = self.extractor(u_raw.detach())
        loss_con = F.mse_loss(theta_pred, theta_raw, reduction='none')
        u_recon = self.generator(theta_pred)
        loss_rec = F.mse_loss(u_recon, u_raw.detach(), reduction='none')
        aux_loss = (alpha_final * (loss_con + loss_rec.mean(dim=1, keepdim=True))).mean()

        log_kc = torch.log(1.0 + kc_counts.float())
        gamma_k = self.gamma.weight
        
        alpha_s = self.local_growth_mod(theta_raw)
        g_local = alpha_s * torch.tanh(log_kc)
        
        h_total = torch.matmul(log_kc, gamma_k) # (B, dim)
        h_self = log_kc.unsqueeze(-1) * gamma_k.unsqueeze(0) # (B, K, dim)
        h_bg = h_total.unsqueeze(1) - h_self # (B, K, dim)
        
        collision = h_bg * gamma_k.unsqueeze(0) # (B, K, dim)
        raw_transfer = self.interference_mlp(collision).squeeze(-1) # (B, K)
        
        beta_s = self.transfer_mod(theta_raw)
        delta_gain = g_local + beta_s * raw_transfer

        h_exp_vec = torch.tanh(self.experience_encoder(log_kc))
        g = torch.sigmoid(self.gain_gate(torch.cat([u_base, h_exp_vec], dim=1)))
        u_final = (1 - g) * u_base + g * h_exp_vec
        
        L_static = torch.matmul(u_final, gamma_k.T)
        L_spec = L_static + delta_gain
        
        dk = torch.sigmoid(self.dk_projector(gamma_k)).squeeze(-1)
        L_base = self.alpha * theta_raw - dk * torch.exp(-self.beta * theta_raw)
        
        e_emb = self.exer(e_id)
        L_diff = torch.matmul(e_emb, gamma_k.T)
        disc_e = torch.sigmoid(self.disc(e_id))
        
        net_input = (L_base + L_spec - L_diff) * disc_e * q_vec
        output = torch.sigmoid(self.prednet_full3(self.drop_2(torch.tanh(self.prednet_full2(self.drop_1(torch.tanh(self.prednet_full1(net_input))))))))

        return output.view(-1), aux_loss

class EADCD(CDM):
    def __init__(self, n, m, k, dim, aux_weight=0.1, emb_dropout=0.3):
        super(EADCD, self).__init__()
        self.net = EADCD_Net(n, m, k, dim, emb_dropout)
        self.aux_weight = aux_weight

    def train(self, train, valid, lr, device, epoch, save_path):
        self.net = self.net.to(device)
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        best_auc = 0
        for i in range(epoch):
            self.net.train()
            epoch_loss = []
            for batch in tqdm(train, f"Epoch {i + 1}/{epoch}"):
                s_id, e_id, k_vec, kc_counts, y = batch 
                s_id, e_id, k_vec, kc_counts, y = s_id.to(device), e_id.to(device), k_vec.to(device), kc_counts.to(device), y.to(device)
                pred, aux_loss = self.net(s_id, e_id, k_vec, kc_counts)
                total_loss = loss_func(pred, y.float()) + self.aux_weight * aux_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_loss.append(total_loss.item())
                with torch.no_grad():
                    self.net.alpha.data.clamp_(min=1e-8)
                    self.net.beta.data.clamp_(min=1e-8)
            if valid is not None:
                auc, _, _, _ = self.eval(valid, device)
                if auc > best_auc:
                    best_auc = auc
                    torch.save(self.net.state_dict(), save_path)
        return best_auc

    def eval(self, test, device):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in tqdm(test, "Evaluating"):
                s_id, e_id, k_vec, kc_counts, y = batch 
                s_id, e_id, k_vec, kc_counts = s_id.to(device), e_id.to(device), k_vec.to(device), kc_counts.to(device)
                pred, _ = self.net(s_id, e_id, k_vec, kc_counts)
                y_true.extend(y.numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))