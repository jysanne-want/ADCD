import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from tqdm import tqdm
from EduCDM import CDM


def _unwrap_dynamo_wrapped_method(method):
    return getattr(method, "__wrapped__", method)


def _patch_torch_optimizer_for_broken_dynamo():
    """
    Work around PyTorch 2.7.x environments where torch._dynamo import fails
    inside optimizer wrappers before the actual training logic starts.
    """
    optimizer_cls = torch.optim.Optimizer
    optimizer_cls.add_param_group = _unwrap_dynamo_wrapped_method(optimizer_cls.add_param_group)
    optimizer_cls.zero_grad = _unwrap_dynamo_wrapped_method(optimizer_cls.zero_grad)
    optimizer_cls.state_dict = _unwrap_dynamo_wrapped_method(optimizer_cls.state_dict)
    optimizer_cls.load_state_dict = _unwrap_dynamo_wrapped_method(optimizer_cls.load_state_dict)

    original_adam_step = _unwrap_dynamo_wrapped_method(torch.optim.Adam.step)

    def _safe_adam_step(self, closure=None):
        with torch.no_grad():
            return original_adam_step(self, closure)

    torch.optim.Adam.step = _safe_adam_step


_patch_torch_optimizer_for_broken_dynamo()


logger = logging.getLogger(__name__)


class PosLinear(nn.Linear):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input_tensor, weight, self.bias)


class EADCD_Net(nn.Module):
    def __init__(self, n, m, k, dim, emb_dropout=0.3):
        super().__init__()
        self.n = n
        self.m = m
        self.k = k
        self.dim = dim
        self.emb_dropout = emb_dropout

        hidden_half = max(1, self.dim // 2)

        self.theta = nn.Embedding(self.n, 1)
        self.delta = nn.Embedding(self.n, self.dim)

        self.generator = nn.Sequential(
            nn.Linear(1, hidden_half),
            nn.Tanh(),
            nn.Linear(hidden_half, self.dim),
        )
        self.extractor = nn.Sequential(
            nn.Linear(self.dim, hidden_half),
            nn.Tanh(),
            nn.Linear(hidden_half, 1),
        )
        self.belief_gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.gamma = nn.Embedding(self.k, self.dim)

        self.local_growth_mod = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )
        self.interference_mlp = nn.Sequential(
            nn.Linear(self.dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.transfer_mod = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.experience_encoder = nn.Linear(self.k, self.dim)
        self.time_context_encoder = nn.Sequential(
            nn.Linear(3, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
        )
        self.gain_gate = nn.Linear(self.dim * 3, self.dim)

        self.exer = nn.Embedding(self.m, self.dim)
        self.disc = nn.Embedding(self.m, 1)
        self.disc_time_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
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
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_normal_(param)

    def _ensure_time_feature(self, feature, batch_size, device):
        if feature is None:
            return torch.zeros(batch_size, 1, device=device)
        if feature.dim() == 1:
            return feature.view(-1, 1).to(device)
        return feature.to(device)

    def _stabilize_scalar_feature(self, feature):
        return torch.sign(feature) * torch.log1p(torch.abs(feature))

    def _split_counts_and_legacy_rt(self, kc_counts_raw):
        if kc_counts_raw.shape[1] == self.k + 1:
            return kc_counts_raw[:, 1:], kc_counts_raw[:, :1]
        if kc_counts_raw.shape[1] == self.k:
            return kc_counts_raw, None
        if kc_counts_raw.shape[1] > self.k + 1:
            return kc_counts_raw[:, -self.k:], kc_counts_raw[:, :1]
        raise ValueError(
            f"Unexpected kc_counts shape: {tuple(kc_counts_raw.shape)}. Expected second dim {self.k} or {self.k + 1}."
        )

    def forward(self, s_id, e_id, q_vec, kc_counts_raw, rt=None, user_speed=None, item_load=None):
        kc_counts, legacy_rt = self._split_counts_and_legacy_rt(kc_counts_raw.float())
        batch_size = s_id.shape[0]
        device = s_id.device

        rt = self._ensure_time_feature(rt, batch_size, device) if rt is not None else legacy_rt
        rt = self._ensure_time_feature(rt, batch_size, device)
        user_speed = self._ensure_time_feature(user_speed, batch_size, device)
        item_load = self._ensure_time_feature(item_load, batch_size, device)

        rt_feat = self._stabilize_scalar_feature(rt.float())
        user_speed_feat = self._stabilize_scalar_feature(user_speed.float())
        item_load_feat = self._stabilize_scalar_feature(item_load.float())

        theta_raw = self.theta(s_id)
        u_raw = self.delta(s_id)
        total_counts = kc_counts.sum(dim=1, keepdim=True)
        log_total = torch.log1p(total_counts)

        u_prior = self.generator(theta_raw)
        belief_input = torch.cat([log_total, user_speed_feat, rt_feat], dim=1)
        alpha = self.belief_gate(belief_input)

        if self.training and self.emb_dropout > 0:
            mask = (torch.rand(batch_size, 1, device=device) > self.emb_dropout).float()
            alpha_final = alpha * mask
        else:
            alpha_final = alpha

        u_base = (1 - alpha_final) * u_prior + alpha_final * u_raw

        theta_pred = self.extractor(u_raw.detach())
        loss_con = F.mse_loss(theta_pred, theta_raw, reduction="none")
        u_recon = self.generator(theta_pred)
        loss_rec = F.mse_loss(u_recon, u_raw.detach(), reduction="none")
        aux_loss = (alpha_final * (loss_con + loss_rec.mean(dim=1, keepdim=True))).mean()

        log_kc = torch.log1p(kc_counts)
        gamma_k = self.gamma.weight

        growth_input = torch.cat([theta_raw, user_speed_feat, rt_feat], dim=1)
        alpha_s = self.local_growth_mod(growth_input)
        g_local = alpha_s * torch.tanh(log_kc)

        h_total = torch.matmul(log_kc, gamma_k)
        h_self = log_kc.unsqueeze(-1) * gamma_k.unsqueeze(0)
        h_bg = h_total.unsqueeze(1) - h_self

        collision = h_bg * gamma_k.unsqueeze(0)
        raw_transfer = self.interference_mlp(collision).squeeze(-1)

        transfer_input = torch.cat([theta_raw, user_speed_feat, item_load_feat, rt_feat], dim=1)
        beta_s = self.transfer_mod(transfer_input)
        delta_gain = g_local + beta_s * raw_transfer

        h_exp_vec = torch.tanh(self.experience_encoder(log_kc))
        time_context = self.time_context_encoder(torch.cat([user_speed_feat, item_load_feat, rt_feat], dim=1))
        gain_input = torch.cat([u_base, h_exp_vec, time_context], dim=1)
        gain_gate = torch.sigmoid(self.gain_gate(gain_input))
        h_time = h_exp_vec + time_context
        u_final = (1 - gain_gate) * u_base + gain_gate * h_time

        l_static = torch.matmul(u_final, gamma_k.T)
        l_spec = l_static + delta_gain

        dk = torch.sigmoid(self.dk_projector(gamma_k)).squeeze(-1)
        alpha_param = torch.clamp(self.alpha, min=1e-8)
        beta_param = torch.clamp(self.beta, min=1e-8)
        l_base = alpha_param * theta_raw - dk.unsqueeze(0) * torch.exp(-beta_param * theta_raw)

        e_emb = self.exer(e_id)
        l_diff = torch.matmul(e_emb, gamma_k.T)
        disc_time = self.disc_time_proj(torch.cat([item_load_feat, rt_feat], dim=1))
        disc_e = torch.sigmoid(self.disc(e_id) + disc_time)

        net_input = (l_base + l_spec - l_diff) * disc_e * q_vec.float()
        hidden = self.drop_1(torch.tanh(self.prednet_full1(net_input)))
        hidden = self.drop_2(torch.tanh(self.prednet_full2(hidden)))
        output = torch.sigmoid(self.prednet_full3(hidden))
        return output.view(-1), aux_loss


class EADCD(CDM):
    def __init__(self, n, m, k, dim, emb_dropout=0.3):
        super().__init__()
        self.net = EADCD_Net(n=n, m=m, k=k, dim=dim, emb_dropout=emb_dropout)

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 8:
            return batch
        if len(batch) == 5:
            s_id, e_id, q_vec, kc_counts, y = batch
            return s_id, e_id, q_vec, kc_counts, None, None, None, y
        raise ValueError(f"Unexpected batch format with {len(batch)} tensors.")

    def train(self, train, valid, lr, device, epoch, save_path):
        self.net = self.net.to(device)
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        best_auc = float("-inf")
        threshold = 0.001

        for i in range(epoch):
            self.net.train()
            epoch_loss = []

            for batch in tqdm(train, desc=f"Epoch {i + 1}/{epoch}"):
                s_id, e_id, q_vec, kc_counts, rt, user_speed, item_load, y = self._unpack_batch(batch)
                s_id = s_id.to(device)
                e_id = e_id.to(device)
                q_vec = q_vec.to(device)
                kc_counts = kc_counts.to(device)
                y = y.to(device)
                rt = None if rt is None else rt.to(device)
                user_speed = None if user_speed is None else user_speed.to(device)
                item_load = None if item_load is None else item_load.to(device)

                pred, aux_loss = self.net(s_id, e_id, q_vec, kc_counts, rt, user_speed, item_load)
                loss = loss_func(pred, y.float()) + aux_loss

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
                s_id, e_id, q_vec, kc_counts, rt, user_speed, item_load, y = self._unpack_batch(batch)
                s_id = s_id.to(device)
                e_id = e_id.to(device)
                q_vec = q_vec.to(device)
                kc_counts = kc_counts.to(device)
                rt = None if rt is None else rt.to(device)
                user_speed = None if user_speed is None else user_speed.to(device)
                item_load = None if item_load is None else item_load.to(device)

                pred, _ = self.net(s_id, e_id, q_vec, kc_counts, rt, user_speed, item_load)
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
