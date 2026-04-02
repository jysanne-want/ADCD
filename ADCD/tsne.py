import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adcd import ADCD 

DATASET_NAME = "Junyi"
SPLIT_MODE = "Random Split"
DATA_DIR = '../Data/data/J123'
MODEL_PATH = 'paras_EADCD/model_random_random_fold_1.snapshot'
EMBEDDING_DIM = 40

OUTPUT_FILE = f'img/J_rand.pdf'

print("Loading model...")
meta_path = os.path.join(DATA_DIR, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)
n, m, k = meta['n'], meta['m'], meta['k']

model = ADCD(n, m, k, EMBEDDING_DIM)
try:
    model.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}")
    exit()
model.net.eval()

print("Extracting embeddings...")
with torch.no_grad():
    theta_params = model.net.theta.weight.data.cpu().numpy().squeeze()
delta_params = model.net.delta_emb.weight.data.cpu().numpy()

print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
delta_2d = tsne.fit_transform(delta_params)

print("Plotting...")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype'] = 42

fig, ax = plt.subplots(figsize=(6, 6))

cmap_name = 'Spectral_r' 

scatter = ax.scatter(
    delta_2d[:, 0], 
    delta_2d[:, 1], 
    c=theta_params, 
    cmap=cmap_name, 
    s=12,
    alpha=0.8,
    edgecolors='none'
)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

for spine in ax.spines.values():
    spine.set_visible(False)

label_text = f"{DATASET_NAME}\n({SPLIT_MODE})"
ax.text(
    0.02, 0.98, label_text,
    transform=ax.transAxes,
    fontsize=16,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle='square,pad=0.4', fc='white', alpha=0.9, ec='black', lw=1)
)

cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.outline.set_visible(False)
cbar.ax.tick_params(size=0)

plt.tight_layout()
os.makedirs('img', exist_ok=True)
plt.savefig(OUTPUT_FILE, bbox_inches='tight', dpi=300)
print(f"Visualization saved to {OUTPUT_FILE}")