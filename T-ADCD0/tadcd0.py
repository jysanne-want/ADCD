import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from EduCDM import CDM
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from tqdm import tqdm


logger = logging.getLogger(__name__)


class PosLinear(nn.Linear):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input_tensor, weight, self.bias)


class ADCD_Net(nn.Module):
    def __init__(self, n, m, k, dim, emb_dropout=0.3):
        super().__init__()
        self.n = n
        self.m = m
        self.k = k
        self.dim = dim
        self.emb_dropout = emb_dropout

        self.theta = nn.Embedding(self.n, 1)
        
        self.stu_emb = nn.Embedding(self.n, self.dim)

        self.student_time_encoder = nn.Sequential(
            nn.Linear(2, self.dim),
            nn.Tanh(),
        )
        self.item_time_encoder = nn.Sequential(
            nn.Linear(2, self.dim),
            nn.Tanh(),
        )

        self.gamma = nn.Embedding(self.k, self.dim)

        self.experience_encoder = nn.Linear(self.k, self.dim)
        self.time_context_encoder = nn.Sequential(
            nn.Linear(5, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
        )
        
        self.fusion_gate = nn.Linear(self.dim * 3, self.dim)
        
        self.dk_projector = nn.Linear(self.dim, 1)
        self.exer = nn.Embedding(self.m, self.dim)
        self.disc = nn.Embedding(self.m, 1)
        self.disc_time_proj = nn.Linear(2, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.prednet_full1 = PosLinear(self.k, 256)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(256, 128)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(128, 1)

        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_normal_(param)

    def _build_time_context(self, rt, user_speed, item_load):
        rt = rt.float()
        user_speed = user_speed.float()
        item_load = item_load.float()
        return torch.cat([
                rt,
                user_speed,
                item_load,
                rt - item_load,
                user_speed - item_load,
            ],
            dim=1,
        )

    def _ensure_time_feature(self, feature, batch_size, device):
        if feature is None:
            return torch.zeros(batch_size, 1, device=device)
        if feature.dim() == 1:
            return feature.view(-1, 1).to(device)
        return feature.to(device)

    def forward(self, s_id, e_id, q_vec, kc_counts, rt=None, user_speed=None, item_load=None):
        gamma_k = self.gamma.weight
        batch_size = s_id.shape[0]
        device = s_id.device

        rt = self._ensure_time_feature(rt, batch_size, device)
        user_speed = self._ensure_time_feature(user_speed, batch_size, device)
        item_load = self._ensure_time_feature(item_load, batch_size, device)

        theta_s = self.theta(s_id)
        
        student_time_input = torch.cat([user_speed.float(), rt.float()], dim=1)
        item_time_input = torch.cat([item_load.float(), rt.float()], dim=1)
        student_time_context = self.student_time_encoder(student_time_input)
        item_time_context = self.item_time_encoder(item_time_input)

        u_base = self.stu_emb(s_id) + student_time_context

        if self.training and self.emb_dropout > 0:
            mask = (torch.rand(s_id.shape[0], 1, device=s_id.device) > self.emb_dropout).float()
            u_base = u_base * mask

        h_exp = torch.tanh(self.experience_encoder(torch.log1p(kc_counts.float())))
        time_context = self.time_context_encoder(self._build_time_context(rt, user_speed, item_load))
        h_time = h_exp + time_context
        
        fusion_input = torch.cat([u_base, h_exp, time_context], dim=1)
        fusion_gate = torch.sigmoid(self.fusion_gate(fusion_input))
        u_final = (1 - fusion_gate) * u_base + fusion_gate * h_time

        dk = torch.sigmoid(self.dk_projector(gamma_k)).squeeze(-1)
        alpha_param = torch.clamp(self.alpha, min=1e-8)
        beta_param = torch.clamp(self.beta, min=1e-8)
        lf = alpha_param * theta_s - dk.unsqueeze(0) * torch.exp(-beta_param * theta_s)
        
        delta_sk = torch.matmul(u_final, gamma_k.T)
        p_sk = torch.sigmoid(lf + delta_sk)

        e = self.exer(e_id) + item_time_context
        d_ek = torch.sigmoid(torch.matmul(e, gamma_k.T))
        disc_e = torch.sigmoid(self.disc(e_id) + self.disc_time_proj(item_time_input))

        input_x = q_vec.float() * (p_sk - d_ek) * disc_e
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        return output.view(-1)


class ADCD(CDM):
    def __init__(self, n, m, k, dim, emb_dropout=0.3):
        super().__init__()
        self.net = ADCD_Net(
            n=n,
            m=m,
            k=k,
            dim=dim,
            emb_dropout=emb_dropout,
        )

    def train(self, train, valid, lr, device, epoch, save_path):
        self.net = self.net.to(device)
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        best_auc = float("-inf")
        threshold = 0.001

        for i in range(epoch):
            self.net.train()
            epoch_loss =[]

            for batch in tqdm(train, desc=f"Epoch {i + 1}/{epoch}"):
                s_id, e_id, k_vec, kc_counts, rt, user_speed, item_load, y = batch
                s_id = s_id.to(device)
                e_id = e_id.to(device)
                k_vec = k_vec.to(device)
                kc_counts = kc_counts.to(device)
                rt = rt.to(device)
                user_speed = user_speed.to(device)
                item_load = item_load.to(device)
                y = y.to(device)

                pred = self.net(s_id, e_id, k_vec, kc_counts, rt, user_speed, item_load)
                loss = loss_func(pred, y.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            if valid is None:
                continue

            auc, acc, rmse, f1 = self.eval(valid, device)
            avg_loss = float(np.mean(epoch_loss)) if epoch_loss else float("nan")
            print(
                f"\n[Epoch {i + 1}] loss={avg_loss:.6f}, "
                f"val_auc={auc:.6f}, val_acc={acc:.6f}, val_rmse={rmse:.6f}, val_f1={f1:.6f}"
            )

            should_save = best_auc == float("-inf") or auc > best_auc + threshold
            if should_save:
                best_auc = auc
                torch.save(self.net.state_dict(), save_path)
                print(f"  [*] Validation improved. Model saved to {save_path}.")
            else:
                print(
                    f"  [!] Validation AUC did not improve beyond {threshold}. "
                    f"Current={auc:.6f}, Best={best_auc:.6f}. Early stopping."
                )
                break

        if valid is None:
            torch.save(self.net.state_dict(), save_path)
            return float("nan")
        return best_auc

    def eval(self, test, device):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in tqdm(test, desc="Evaluating"):
                s_id, e_id, k_vec, kc_counts, rt, user_speed, item_load, y = batch
                s_id = s_id.to(device)
                e_id = e_id.to(device)
                k_vec = k_vec.to(device)
                kc_counts = kc_counts.to(device)
                rt = rt.to(device)
                user_speed = user_speed.to(device)
                item_load = item_load.to(device)

                pred = self.net(s_id, e_id, k_vec, kc_counts, rt, user_speed, item_load)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        y_label = y_pred_np >= 0.5
        return (
            roc_auc_score(y_true_np, y_pred_np),
            accuracy_score(y_true_np, y_label),
            np.sqrt(mean_squared_error(y_true_np, y_pred_np)),
            f1_score(y_true_np, y_label),
        )

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location="cpu"))