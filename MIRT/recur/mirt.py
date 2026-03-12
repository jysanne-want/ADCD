import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM

def irt2pl(theta, a, b, F=torch):
    return 1 / (1 + F.exp(- F.sum(a * theta, dim=-1) + b.squeeze()))


class MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, a_range=None):
        super(MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.a_range = a_range
        
        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.a = nn.Embedding(self.item_num, latent_dim)
        self.b = nn.Embedding(self.item_num, 1)

        nn.init.xavier_normal_(self.theta.weight)
        nn.init.xavier_normal_(self.a.weight)
        nn.init.xavier_normal_(self.b.weight)

    def forward(self, user, item):
        theta = self.theta(user)
        a = self.a(item)
        
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a) 
            
        b = self.b(item) 
        
        return irt2pl(theta, a, b)


class MIRT(CDM):
    def __init__(self, n, m, k, **kwargs):
        super(MIRT, self).__init__()
        dim = kwargs.get('dim', 16) 
        a_range = kwargs.get('a_range', None)
        
        self.irt_net = MIRTNet(user_num=n, item_num=m, latent_dim=dim, a_range=a_range)

    def train(self, train_loader, valid_loader, lr, device, epochs, save_path):
        logging.info(f"Training MIRT (dim={self.irt_net.latent_dim})... (lr={lr})")
        self.irt_net = self.irt_net.to(device)
        
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.irt_net.parameters(), lr=lr)

        best_auc = 0
        best_epoch = 0
        threshold = 0.001

        for epoch_i in range(epochs):
            self.irt_net.train()
            epoch_losses = []
            
            for batch_data in tqdm(train_loader, f"Epoch {epoch_i + 1}/{epochs}"):
                user_id, item_id, _, _, y = batch_data
                
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                y = y.to(device)
                
                pred = self.irt_net(user_id, item_id)
                
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
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, "Evaluating"):
                user_id, item_id, _, _, y = batch_data
                
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                
                pred = self.irt_net(user_id, item_id)
                
                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(y.tolist())

        y_pred_binary = (np.array(y_pred) >= 0.5).astype(int)
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred_binary)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        f1 = f1_score(y_true, y_pred_binary)
        return auc, acc, rmse, f1

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)