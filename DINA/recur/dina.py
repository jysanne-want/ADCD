import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM



class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class DINANet(nn.Module):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(DINANet, self).__init__()
        self._user_num = user_num
        self._item_num = item_num
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess

        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)

    def forward(self, user, item, knowledge, *args):
        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        
        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100, 1e-6)
            self.step = self.step + 1 if self.step < self.max_step else 0
            
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            return (1 - slip) ** n * guess ** (1 - n)


class STEDINANet(DINANet):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(STEDINANet, self).__init__(user_num, item_num, hidden_dim, max_slip, max_guess, *args, **kwargs)
        self.sign = StraightThroughEstimator()

    def forward(self, user, item, knowledge, *args):
        theta = self.sign(self.theta(user))
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        
        mask_theta = (knowledge == 0) + (knowledge == 1) * theta
        n = torch.prod((mask_theta + 1) / 2, dim=-1)
        
        return torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)



class DINA(CDM):
    def __init__(self, n, m, k, **kwargs):
        super(DINA, self).__init__()
        
        hidden_dim = k
        
        ste = kwargs.get('ste', True)
        
        max_slip = kwargs.get('max_slip', 0.4)
        max_guess = kwargs.get('max_guess', 0.4)

        if ste:
            self.dina_net = STEDINANet(n, m, hidden_dim, max_slip=max_slip, max_guess=max_guess)
        else:
            self.dina_net = DINANet(n, m, hidden_dim, max_slip=max_slip, max_guess=max_guess)

    def train(self, train_loader, valid_loader, lr, device, epochs, save_path):
        logging.info(f"Training DINA (k={self.dina_net.theta.embedding_dim}, STE={isinstance(self.dina_net, STEDINANet)})... (lr={lr})")
        self.dina_net = self.dina_net.to(device)
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dina_net.parameters(), lr=lr)

        best_auc = 0
        best_epoch = 0
        threshold = 0.001

        for epoch_i in range(epochs):
            self.dina_net.train()
            epoch_losses = []
            
            for batch_data in tqdm(train_loader, f"Epoch {epoch_i + 1}/{epochs}"):
                user_id, item_id, knowledge, _, y = batch_data
                
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge = knowledge.to(device)
                y = y.to(device)
                
                predicted_response = self.dina_net(user_id, item_id, knowledge)
                
                loss = loss_function(predicted_response, y.float().view(-1))
                
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
        self.dina_net = self.dina_net.to(device)
        self.dina_net.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, "Evaluating"):
                user_id, item_id, knowledge, _, y = batch_data
                
                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge = knowledge.to(device)
                
                pred = self.dina_net(user_id, item_id, knowledge)
                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(y.tolist())

        y_pred_binary = (np.array(y_pred) >= 0.5).astype(int)
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred_binary)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        f1 = f1_score(y_true, y_pred_binary)
        
        self.dina_net.train()
        return auc, acc, rmse, f1

    def save(self, filepath):
        torch.save(self.dina_net.state_dict(), filepath)
        logging.info("Save parameters to %s" % filepath)

    def load(self, filepath):
        self.dina_net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("Load parameters from %s" % filepath)