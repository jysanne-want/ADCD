import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
from EduCDM import CDM
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class CACD_Net(nn.Module):
    def __init__(self, n, m, k, affect_dim=4):
        super(CACD_Net, self).__init__()
        self.n = n 
        self.m = m 
        self.k = k 
        self.affect_dim = affect_dim 

        self.student_emb = nn.Embedding(self.n, self.k)       
        self.student_affect = nn.Embedding(self.n, self.affect_dim) 
        self.k_difficulty = nn.Embedding(self.m, self.k)    
        self.e_discrimination = nn.Embedding(self.m, 1)     

        self.affect_perception = nn.Linear(self.affect_dim + self.k, self.affect_dim)
        
        self.guess_layer = nn.Linear(self.affect_dim, 1)
        self.slip_layer = nn.Linear(self.affect_dim, 1)

        self.prednet_len1, self.prednet_len2 = 256, 128
        self.diagnosis_full1 = PosLinear(self.k, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.diagnosis_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.diagnosis_full3 = PosLinear(self.prednet_len2, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, s_id, e_id, q_vec, labels=None):
        stu_emb = torch.sigmoid(self.student_emb(s_id))       
        stu_affect_trait = torch.sigmoid(self.student_affect(s_id)) 
        k_diff = torch.sigmoid(self.k_difficulty(e_id))       
        e_disc = torch.sigmoid(self.e_discrimination(e_id)) * 10 

        affect_input = torch.cat((stu_affect_trait, k_diff), dim=1)
        affect_vec = torch.sigmoid(self.affect_perception(affect_input))

        diag_input = e_disc * (stu_emb - k_diff) * q_vec
        input_x = torch.sigmoid(self.diagnosis_full1(diag_input))
        input_x = self.drop_1(input_x)
        input_x = torch.sigmoid(self.diagnosis_full2(input_x))
        input_x = self.drop_2(input_x)
        y_star = torch.sigmoid(self.diagnosis_full3(input_x)) 

        g = torch.sigmoid(self.guess_layer(affect_vec)) 
        s = torch.sigmoid(self.slip_layer(affect_vec))  
        
        y_hat = ((1 - s) * y_star) + (g * (1 - y_star))
        y_hat = y_hat.view(-1)

        if labels is not None:
            closs = self.contrastive_loss(affect_vec, labels)
            return y_hat, closs
        
        return y_hat

    def contrastive_loss(self, affect, label):
        tau = 0.1 
        batch_size = affect.shape[0]
        
        norm_affect = F.normalize(affect, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_affect, norm_affect.T) # [B, B]
        
        label = label.view(-1, 1)
        mask_positive = torch.eq(label, label.T).float().to(affect.device)
        mask_negative = 1 - mask_positive
        
        mask_0 = (1 - torch.eye(batch_size)).to(affect.device)
        mask_positive = mask_positive * mask_0
        
        exp_sim = torch.exp(similarity_matrix / tau) * mask_0
        
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        log_prob = torch.log((exp_sim / (sum_exp_sim + 1e-10)) + 1e-10)
        
        pos_counts = torch.sum(mask_positive, dim=1)
        loss = -torch.sum(log_prob * mask_positive, dim=1) / (pos_counts + 1e-10)
        
        return torch.mean(loss)

class CACD(CDM):
    def __init__(self, n, m, k, affect_dim=4):
        super(CACD, self).__init__()
        self.net = CACD_Net(n, m, k, affect_dim)

    def train(self, train, valid, lr, device, epoch, save_path, lambda_ca=1.0, patience=3):

        self.net = self.net.to(device)
        diagnosis_loss_func = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        best_auc = 0
        best_epoch = 0
        patience_counter = 0  
        threshold = 0.001   

        for i in range(epoch):
            self.net.train()
            epoch_diag_loss = []
            epoch_closs = []
            
            for batch in tqdm(train, f"Epoch {i + 1}/{epoch}"):
                s_id, e_id, k_vec, y = batch
                s_id, e_id, k_vec, y = s_id.to(device), e_id.to(device), k_vec.to(device), y.to(device)
                
                pred, closs = self.net(s_id, e_id, k_vec, labels=y)
                diag_loss = diagnosis_loss_func(pred, y.float())
                total_loss = diag_loss + lambda_ca * closs

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_diag_loss.append(diag_loss.item())
                epoch_closs.append(closs.item())

            avg_diag = np.mean(epoch_diag_loss)
            logging.info(f"[Epoch {i + 1}] Diag Loss: {avg_diag:.4f}")

            if valid is not None:
                auc, acc, rmse, f1 = self.eval(valid, device)
                logging.info(f"[Epoch {i + 1}] Validation AUC: {auc:.4f}")

                if auc > (best_auc + threshold):
                    best_auc = auc
                    best_epoch = i + 1
                    self.save(save_path)
                    logging.info(f"Significant improvement! Best model updated (AUC: {best_auc:.4f})")
                    patience_counter = 0  
                else:
                    patience_counter += 1
                    logging.info(f"No significant improvement (>0.001) for {patience_counter} epochs.")
                    
                    if patience_counter >= patience:
                        logging.info(f"Early stopping triggered. Training finished at epoch {i + 1}.")
                        break

        return best_epoch
    
    def eval(self, test, device):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in tqdm(test, "Evaluating"):
                s_id, e_id, k_vec, y = batch
                s_id, e_id, k_vec = s_id.to(device), e_id.to(device), k_vec.to(device)

                pred = self.net(s_id, e_id, k_vec)

                y_true.extend(y.numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_pred_bin = (y_pred >= 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred_bin)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        f1 = f1_score(y_true, y_pred_bin)

        return auc, acc, rmse, f1

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info(f"Parameters saved to {filepath}")

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info(f"Parameters loaded from {filepath}")