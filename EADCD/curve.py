import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from eadcd import EADCD

MODEL_PATH = 'paras_EADCD/model_real_real_split.snapshot' 
DATA_DIR = '../Data/data/N123'
EMBEDDING_DIM = 40
MAX_SIMULATE_COUNTS = 50 

print("Loading model...")
meta_path = os.path.join(DATA_DIR, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)
n, m, k = meta['n'], meta['m'], meta['k']

model = EADCD(n, m, k, dim=EMBEDDING_DIM)
model.load(MODEL_PATH)
model.net.eval()

def find_best_target_by_probing():
    with torch.no_grad():
        gamma = model.net.gamma.weight
        sim_count = torch.tensor(10.0, device=gamma.device)
        log_c = torch.log(1.0 + sim_count)
        collision_matrix = log_c * gamma.unsqueeze(1) * gamma.unsqueeze(0)
        collision_flat = collision_matrix.view(k * k, EMBEDDING_DIM)
        transfer_flat = model.net.interference_mlp(collision_flat).squeeze(-1)
        transfer_matrix = transfer_flat.view(k, k).cpu().numpy()
        np.fill_diagonal(transfer_matrix, np.nan)
        max_transfer = np.nanmax(transfer_matrix, axis=1)
        min_transfer = np.nanmin(transfer_matrix, axis=1)
        score = max_transfer - min_transfer
        valid_mask = (max_transfer > 0.05) & (min_transfer < -0.05)
        
        if valid_mask.any():
            score[~valid_mask] = -np.inf
        target_kc = np.argmax(score)

        best_helper_kc = np.nanargmax(transfer_matrix[target_kc])
        worst_enemy_kc = np.nanargmin(transfer_matrix[target_kc])
        helper_gain = transfer_matrix[target_kc, best_helper_kc]
        enemy_gain = transfer_matrix[target_kc, worst_enemy_kc]
        gamma_np = gamma.cpu().numpy()
        sims = cosine_similarity(gamma_np[target_kc].reshape(1, -1), gamma_np)[0]
        helper_sim = sims[best_helper_kc]
        enemy_sim = sims[worst_enemy_kc]
        return target_kc, best_helper_kc, worst_enemy_kc, helper_gain, enemy_gain, helper_sim, enemy_sim

def simulate_gain(student_idx, target_kc, practice_kc, practice_counts_range):
    with torch.no_grad():
        s_tensor = torch.tensor([student_idx]).long()
        theta = model.net.theta(s_tensor)
        
        alpha_s = model.net.local_growth_mod(theta) 
        beta_s = model.net.transfer_mod(theta)      
        
        gamma = model.net.gamma.weight 
        gamma_target = gamma[target_kc].unsqueeze(0) 
        
        gains =[]
        
        for count in practice_counts_range:
            kc_counts = torch.zeros(1, k)
            kc_counts[0, practice_kc] = count
            
            log_kc_counts = torch.log(1.0 + kc_counts)
            
            g_local = alpha_s * torch.tanh(log_kc_counts[:, target_kc])
            
            h_total = log_kc_counts[:, practice_kc].unsqueeze(1) * gamma[practice_kc].unsqueeze(0)
            
            if practice_kc == target_kc:
                h_background = torch.zeros_like(h_total)
            else:
                h_background = h_total
            
            collision = h_background * gamma_target 
            raw_transfer = model.net.interference_mlp(collision).squeeze(-1)
            
            total_gain = g_local + beta_s * raw_transfer
            gains.append(total_gain.item())
            
        return np.array(gains)

target_kc, helper_kc, enemy_kc, helper_gain, enemy_gain, helper_sim, enemy_sim = find_best_target_by_probing()

with torch.no_grad():
    thetas = model.net.theta.weight.data.cpu().numpy().squeeze()
    s_high = np.argsort(thetas)[-10] 
    s_low = np.argsort(thetas)[10]

print(f"Visualization Setup:")
print(f"  Target KC: {target_kc}")
print(f"  Helper KC: {helper_kc} (Transfer Val: {helper_gain:.3f}, Semantic Sim: {helper_sim:.2f}) -> Should go UP")
print(f"  Enemy  KC: {enemy_kc} (Interference Val: {enemy_gain:.3f}, Semantic Sim: {enemy_sim:.2f}) -> Should go DOWN")

x_axis = np.arange(MAX_SIMULATE_COUNTS)

def format_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)

plt.figure(figsize=(6, 5))
ax1 = plt.gca()
y_high_1 = simulate_gain(s_high, target_kc, target_kc, x_axis)
y_low_1 = simulate_gain(s_low, target_kc, target_kc, x_axis)

ax1.plot(x_axis, y_high_1, color='#2166ac', linewidth=2.5, label='High $\\theta$ Student')
ax1.plot(x_axis, y_low_1, color='#b2182b', linewidth=2.5, label='Low $\\theta$ Student')
ax1.set_xlabel('Practice Counts', fontsize=12)
ax1.set_ylabel('Proficiency Gain', fontsize=12)
ax1.legend()
format_axes(ax1)
plt.tight_layout()
plt.savefig('lcurve_a.pdf', dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
ax2 = plt.gca()
y_high_2 = simulate_gain(s_high, target_kc, helper_kc, x_axis)
y_low_2 = simulate_gain(s_low, target_kc, helper_kc, x_axis)

ax2.plot(x_axis, y_high_2, color='#2166ac', linewidth=2.5, label='High $\\theta$ Student')
ax2.plot(x_axis, y_low_2, color='#b2182b', linewidth=2.5, label='Low $\\theta$ Student')
ax2.set_xlabel(f'Practice Counts on KC {helper_kc}', fontsize=12)
ax2.set_ylabel('Proficiency Gain', fontsize=12)
ax2.axhline(0, color='black', linestyle='--', alpha=0.5) 
ax2.legend()
format_axes(ax2)
plt.tight_layout()
plt.savefig('lcurve_b.pdf', dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
ax3 = plt.gca()
y_high_3 = simulate_gain(s_high, target_kc, enemy_kc, x_axis)
y_low_3 = simulate_gain(s_low, target_kc, enemy_kc, x_axis)

ax3.plot(x_axis, y_high_3, color='#2166ac', linewidth=2.5, label='High $\\theta$ Student')
ax3.plot(x_axis, y_low_3, color='#b2182b', linewidth=2.5, label='Low $\\theta$ Student')
ax3.set_xlabel(f'Practice Counts on KC {enemy_kc}', fontsize=12)
ax3.set_ylabel('Proficiency Gain', fontsize=12)
ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
ax3.legend()
format_axes(ax3)
plt.tight_layout()
plt.savefig('lcurve_c.pdf', dpi=300)
plt.close()
