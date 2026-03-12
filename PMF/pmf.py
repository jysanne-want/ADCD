import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM


class PMFNet(nn.Module):
    def __init__(self, user_num, item_num, dim):
        super(PMFNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dim = dim

        self.user_emb = nn.Embedding(self.user_num, self.dim)
        self.item_emb = nn.Embedding(self.item_num, self.dim)
        
        self.user_bias = nn.Embedding(self.user_num, 1)
        self.item_bias = nn.Embedding(self.item_num, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)

    def forward(self, user_id, item_id):
        u = self.user_emb(user_id)
        i = self.item_emb(item_id)
        
        interaction = (u * i).sum(dim=1, keepdim=True)
        
        u_b = self.user_bias(user_id)
        i_b = self.item_bias(item_id)
        logits = interaction + u_b + i_b + self.global_bias
        
        return torch.sigmoid(logits).view(-1)


class PMF(CDM):
    def __init__(self, n, m, k, **kwargs):
        super(PMF, self).__init__()
        dim = kwargs.get('dim', 64) 
        
        self.net = PMFNet(user_num=n, item_num=m, dim=dim)

    def train(self, train_loader, valid_loader, lr, device, epochs, save_path):
        logging.info(f"Training PMF (dim={self.net.dim})... (lr={lr})")
        self.net = self.net.to(device)
        loss_function = nn.BCELoss()
        
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)

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