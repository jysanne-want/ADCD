import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adcd import ADCD 

# ==================== 1. 配置参数 ====================
# [在这里修改你要画的数据集和场景]
DATASET_NAME = "Junyi"  # 用于图上显示的文字
SPLIT_MODE = "Random Split"       # 用于图上显示的文字
DATA_DIR = '../Data/data/J123'    # 数据路径
MODEL_PATH = 'paras_EADCD/model_random_random_fold_1.snapshot' # 模型路径
EMBEDDING_DIM = 40

OUTPUT_FILE = f'img/J_rand.pdf'


# ==================== 2. 加载模型 ====================
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

# ==================== 3. 提取特征 ====================
print("Extracting embeddings...")
with torch.no_grad():
    # 提取通用能力 (Gf)
    theta_params = model.net.theta.weight.data.cpu().numpy().squeeze()
    # 提取特异性偏置 (Gc - Specialist)
    # [修正] 使用新的变量名 delta_emb
delta_params = model.net.delta_emb.weight.data.cpu().numpy()


# ==================== 4. t-SNE 降维 ====================
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
delta_2d = tsne.fit_transform(delta_params)

# ==================== 5. 绘图 (高对比度期刊风格) ====================
print("Plotting...")

# 设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype'] = 42

# 创建画布
fig, ax = plt.subplots(figsize=(6, 6))

# --- [修改1] 高对比度配色 ---
# 使用 'Spectral_r' 或 'RdYlBu_r'。
# 这些色谱从深蓝(低)过渡到深红(高)，中间经过黄色/绿色，对比度极高
# 也可以尝试 'jet' (彩虹色)，但在学术界 'Spectral' 更受欢迎
cmap_name = 'Spectral_r' 

scatter = ax.scatter(
    delta_2d[:, 0], 
    delta_2d[:, 1], 
    c=theta_params, 
    cmap=cmap_name, 
    s=12,          # 点的大小
    alpha=0.8,     # 透明度，稍微不透明一点以增加鲜艳度
    edgecolors='none' # 去掉描边，让颜色更纯粹
)

# --- [修改2] 移除四周文字和边框 ---
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# 移除四周的脊柱（边框）
for spine in ax.spines.values():
    spine.set_visible(False)

# --- [修改3] 左上角添加标识 ---
# transform=ax.transAxes 确保坐标是相对画布的 (0~1)
# boxstyle='square' 给文字加一个白底背景，防止遮挡数据点
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

# 添加颜色条 (可选：如果你是把多张图拼在一起，可以只给最后一张图加Colorbar)
# 这里为了完整性保留，您可以根据排版需要注释掉
cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.outline.set_visible(False) # 去掉颜色条边框
cbar.ax.tick_params(size=0)     # 去掉颜色条刻度线

# 保存
plt.tight_layout()
os.makedirs('img', exist_ok=True)
plt.savefig(OUTPUT_FILE, bbox_inches='tight', dpi=300)
print(f"Visualization saved to {OUTPUT_FILE}")