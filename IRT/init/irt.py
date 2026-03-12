import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM


class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, dim=1):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dim = dim
        
        # IRT 参数: theta (能力), a (区分度), b (难度), c (猜测系数)
        self.theta = nn.Embedding(self.user_num, self.dim)
        self.a = nn.Embedding(self.item_num, self.dim)
        self.b = nn.Embedding(self.item_num, self.dim)
        self.c = nn.Embedding(self.item_num, 1)

        # 权重初始化 (模拟原代码的分布设定)
        nn.init.xavier_normal_(self.theta.weight)
        nn.init.xavier_normal_(self.b.weight)
        
        # a (区分度) 通常初始化为正数，这里用 1.0 附近
        nn.init.normal_(self.a.weight, mean=1.0, std=0.1)
        
        # c (猜测) 初始化为较小的值
        nn.init.constant_(self.c.weight, 0.0)

    def forward(self, user_id, item_id):
        theta = self.theta(user_id)  # [batch, dim]
        a = self.a(item_id)          # [batch, dim]
        b = self.b(item_id)          # [batch, dim]
        c = torch.sigmoid(self.c(item_id)) # 限制在 [0, 1] 区间

        D = 1.702
        
        # 3PL-IRT 公式 (Multidimensional)
        # P = c + (1 - c) * sigmoid(D * a * (theta - b))
        # 对应原代码: np.sum(a * (theta - b), axis=-1)
        input_x = (a * (theta - b)).sum(dim=1, keepdim=True)
        prob = c + (1 - c) * torch.sigmoid(D * input_x)
        
        return prob.view(-1)


class IRT(CDM):
    def __init__(self, n, m, k, **kwargs):
        super(IRT, self).__init__()
        # ADCD_Runner 传入: n(用户数), m(题目数), k(知识点数)
        # IRT 通常不需要显式的知识点矩阵(Q-matrix)，而是使用隐向量 dim
        # 这里的 k 参数在 IRT 中用不到，我们通过 kwargs 获取 dim，默认为 1
        dim = kwargs.get('dim', 1) 
        
        self.net = IRTNet(user_num=n, item_num=m, dim=dim)

    def train(self, train_loader, valid_loader, lr, device, epochs, save_path):
        logging.info(f"Training IRT (dim={self.net.dim})... (lr={lr})")
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
                # ================= 适配 Runner 的数据解包 =================
                # Runner 返回: user_idx, item_idx, knowledge_emb, counts, label
                # IRT 不需要 knowledge_emb，因此用 _ 忽略
                user_id, item_id, _, _, y = batch_data
                
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                y = y.to(device)
                
                pred = self.net(user_id, item_id)
                
                loss = loss_function(pred, y.float().view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            logging.info(f"[Epoch {epoch_i + 1}] Average Loss: {np.mean(epoch_losses):.6f}")

            if valid_loader is not None:
                auc, acc, rmse, f1 = self.eval(valid_loader, device=device)
                logging.info(f"[Epoch {epoch_i + 1}] Validation AUC: {auc:.6f}, ACC: {acc:.6f}")

                # ================= 早停逻辑 =================
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
                # ================= 适配数据解包 =================
                user_id, item_id, _, _, y = batch_data
                
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                
                pred = self.net(user_id, item_id)
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
        logging.info("Load parameters from %s" % filepath)