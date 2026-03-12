import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM

# ================= 1. PosLinear 保持不变 =================
class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 权重裁剪，保持非负性（单调性约束）
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


# ================= 2. KaNCD_Net (使用你新提供的正确逻辑) =================
class KaNCD_Net(nn.Module):
    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128

        super(KaNCD_Net, self).__init__()

        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # 初始化不同 mf_type 对应的层
        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        # 初始化权重
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(input_exercise)
        batch, dim = stu_emb.size()
        
        # 扩展维度以匹配知识点数量
        stu_emb_expanded = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb_expanded = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        exer_emb_expanded = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)

        # --- 核心区别：这里使用了新代码中正确的 mf_type 分支逻辑 ---
        if self.mf_type == 'mf':
            stat_emb = torch.sigmoid((stu_emb_expanded * knowledge_emb_expanded).sum(dim=-1, keepdim=False))
            k_difficulty = torch.sigmoid((exer_emb_expanded * knowledge_emb_expanded).sum(dim=-1, keepdim=False))
        
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb_expanded * knowledge_emb_expanded)).view(batch, -1)
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb_expanded * knowledge_emb_expanded)).view(batch, -1)
        
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb_expanded, knowledge_emb_expanded), dim=-1))).view(batch, -1)
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb_expanded, knowledge_emb_expanded), dim=-1))).view(batch, -1)
        
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb_expanded, knowledge_emb_expanded), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb_expanded, knowledge_emb_expanded), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        else:
            # 默认回退到 GMF 逻辑或报错，这里给个默认 GMF 行为作为 fallback
            stat_emb = torch.sigmoid(self.stat_full(stu_emb_expanded * knowledge_emb_expanded)).view(batch, -1)
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb_expanded * knowledge_emb_expanded)).view(batch, -1)

        # 区分度
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        # 预测层
        input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


# ================= 3. KaNCD 类 (适配 ADCD_Runner 和早停) =================
class KaNCD(CDM):
    def __init__(self, n, m, k, **kwargs):
        super(KaNCD, self).__init__()
        # 兼容 Runner 传参
        mf_type = kwargs.get('mf_type', 'gmf')
        dim = kwargs.get('dim', 64)

        self.net = KaNCD_Net(
            student_n=n, 
            exer_n=m, 
            knowledge_n=k, 
            mf_type=mf_type, 
            dim=dim
        )

    def train(self, train_loader, valid_loader, lr, device, epochs, save_path):
        logging.info(f"Training KaNCD ({self.net.mf_type})... (lr={lr})")
        self.net = self.net.to(device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        best_auc = 0
        best_epoch = 0
        threshold = 0.001

        for epoch_i in range(epochs):
            self.net.train()
            epoch_losses = []
            
            for batch_data in tqdm(train_loader, f"Epoch {epoch_i + 1}/{epochs}"):
                # 适配 ADCD_Runner 的 5 个返回值
                # 顺序: user_idx, item_idx, knowledge_emb, ignored_counts, label
                user_info, item_info, knowledge_emb, _, y = batch_data
                
                user_info = user_info.to(device)
                item_info = item_info.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = y.to(device)
                
                pred = self.net(user_info, item_info, knowledge_emb)
                
                # 确保 y 的形状匹配 pred
                loss = loss_function(pred, y.float().view(-1)) 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            logging.info(f"[Epoch {epoch_i + 1}] Average Loss: {np.mean(epoch_losses):.6f}")

            if valid_loader is not None:
                auc, acc, rmse, f1 = self.eval(valid_loader, device=device)
                logging.info(f"[Epoch {epoch_i + 1}] Validation AUC: {auc:.6f}, ACC: {acc:.6f}")

                improvement = auc - best_auc
                if improvement > threshold:
                    best_auc = auc
                    best_epoch = epoch_i + 1
                    self.save(save_path) 
                    logging.info(f"Valid improvement ({improvement:.6f} > {threshold}). Model saved.")
                else:
                    logging.info(f"Early stopping triggered at epoch {epoch_i + 1}.")
                    break

        return best_epoch
    
    def eval(self, test_loader, device):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_data in tqdm(test_loader, "Evaluating"): 
                # 同样的 5 参数解包
                user_id, item_id, knowledge_emb, _, y = batch_data
                
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                
                pred = self.net(user_id, item_id, knowledge_emb)
                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(y.tolist())

        y_pred_binary = (np.array(y_pred) >= 0.5).astype(int)
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred_binary)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        f1 = f1_score(y_true, y_pred_binary)
        return auc, acc, rmse, f1
    
    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("Save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)