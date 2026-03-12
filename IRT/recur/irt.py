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
        
        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)
        nn.init.normal_(self.theta.weight, mean=0, std=1)
        nn.init.normal_(self.b.weight, mean=0, std=1)
        nn.init.normal_(self.a.weight, mean=1, std=0.1)
        nn.init.constant_(self.c.weight, 0.0)

    def forward(self, user_id, item_id):
        # 提取嵌入
        theta = self.theta(user_id).squeeze(-1)  # [batch_size]
        a = self.a(item_id).squeeze(-1)          # [batch_size]
        b = self.b(item_id).squeeze(-1)          # [batch_size]
        c = torch.sigmoid(self.c(item_id)).squeeze(-1) # 限制 c 在 [0, 1]

        D = 1.702
        
        logits = D * a * (theta - b)
        prob = c + (1 - c) * torch.sigmoid(logits)
        
        return prob


class IRT(CDM):
    def __init__(self, n, m, k, **kwargs):
        super(IRT, self).__init__()
        self.net = IRTNet(user_num=n, item_num=m)

    def train(self, train_loader, valid_loader, lr, device, epochs, save_path):
        logging.info(f"Training Neural IRT (PyTorch version)... (lr={lr})")
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
                user_id, item_id, _, _, y = batch_data
                
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                y = y.to(device)
                
                pred = self.net(user_id, item_id)
                
                loss = loss_function(pred, y.float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            logging.info(f"[Epoch {epoch_i + 1}] Average Loss: {avg_loss:.6f}")

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