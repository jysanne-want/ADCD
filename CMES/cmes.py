import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Net
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
import logging
from tqdm import tqdm

class CMESWrapper:
    def __init__(self, n, m, k, topk, **kwargs):
        self.net = Net(student_n=n, exer_n=m, knowledge_n=k, topk=topk)
        self.topk = topk
        self.loss_function = nn.NLLLoss()
        
    def train(self, train_loader, val_loader, lr, device, epochs, save_path, patience=3, min_delta=0.001):
        self.net.to(device)
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        best_auc = 0.0
        best_epoch = -1
        no_improve_cnt = 0 

        for epoch in range(epochs):
            self.net.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
            for batch in pbar:
                stu_ids, pos_ids, kn_embs, labels, neg_ids = [x.to(device) for x in batch]
                optimizer.zero_grad()

                output_pos = self.net(stu_ids, pos_ids, kn_embs)
                output_0 = torch.ones(output_pos.size()).to(device) - output_pos
                output_cat = torch.cat((output_0, output_pos), 1)
                pos_loss = self.loss_function(torch.log(output_cat + 1e-10), labels.long())

                output_negs, neg_scores, bpr_losses = self.net(
                    stu_ids, neg_ids, kn_embs, pos_ids, labels.long()
                )

                all_neg_loss = 0
                for i in range(output_negs.size(1)):
                    each_neg_prob = output_negs[:, i].view(-1, 1)
                    each_neg_score = neg_scores[:, i].squeeze()
                    output_0_neg = torch.ones(each_neg_prob.size()).to(device) - each_neg_prob
                    output_neg_cat = torch.cat((output_0_neg, each_neg_prob), 1)
                    neg_loss = self.loss_function(torch.log(output_neg_cat + 1e-10), each_neg_score.long())
                    all_neg_loss += neg_loss

                total_loss = pos_loss + (all_neg_loss / self.topk) + 0.1 * bpr_losses
                total_loss.backward()
                optimizer.step()
                self.net.apply_clipper()
                
                curr_loss = total_loss.item()
                running_loss += curr_loss
                
                pbar.set_postfix(loss=f"{curr_loss:.4f}")

            auc, acc, rmse, f1 = self.eval(val_loader, device)
            logging.info(f"Epoch {epoch+1} Val Result - AUC: {auc:.4f}, ACC: {acc:.4f}, RMSE: {rmse:.4f}")

            if auc >= best_auc + min_delta:
                logging.info(f"Significant improvement detected ({auc:.4f} >= {best_auc:.4f} + {min_delta})")
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1
                logging.info(f"No significant improvement. Patience: {no_improve_cnt}/{patience}")

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch + 1
                self.save(save_path)
                logging.info(f"*** Best Model Updated (AUC: {best_auc:.4f}) ***")
            
            if no_improve_cnt >= patience:
                logging.info(f"Early stopping triggered! No significant improvement for {patience} epochs.")
                break

        return best_epoch

    def eval(self, loader, device):
 
        self.net.eval()
        self.net.to(device)
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in loader:
                batch = [x.to(device) for x in batch]
                
                if len(batch) == 5:
                    stu_ids, pos_ids, kn_embs, _, labels = batch
                elif len(batch) == 4:
                    stu_ids, pos_ids, kn_embs, labels = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                output = self.net(stu_ids, pos_ids, kn_embs)
                
                output = output.view(-1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(output.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        auc = roc_auc_score(y_true, y_pred)
        y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
        acc = accuracy_score(y_true, y_pred_binary)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        f1 = f1_score(y_true, y_pred_binary)
        
        return auc, acc, rmse, f1

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))