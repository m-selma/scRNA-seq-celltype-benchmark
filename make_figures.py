"""
make_figures.py
---------------
Generate all six publication figures from the benchmark results.

Figures:
    1. Model comparison (accuracy / balanced accuracy / macro F1)
    2. Accuracy-vs-macro-F1 gap
    3. Per-class F1 heatmap
    4. Confusion matrix (best model)
    5. Per-class F1 vs class frequency (with trend)
    6. UMAP embedding colored by cell type

Run after benchmark.py has produced results/.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json
import matplotlib.pyplot as plt
import matplotlib as mpl
import scanpy as sc

mpl.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'font.family': 'sans-serif',
})
ACCENT='#2563eb'; ACCENT2='#dc2626'; GREY='#94a3b8'

df = pd.read_csv('results/model_comparison.csv')
pc = pd.read_csv('results/per_class_f1.csv')
meta = json.load(open('results/meta.json'))

COUNTS = {'CD4 T cells':1144,'CD14+ Monocytes':480,'B cells':342,'CD8 T cells':316,
          'NK cells':154,'FCGR3A+ Monocytes':150,'Dendritic cells':37,'Megakaryocytes':15}
CLASSES = list(COUNTS.keys())

# ---- Figure 1: model comparison ----
fig, ax = plt.subplots(figsize=(9,5.5))
d = df.sort_values('Macro F1')
y = np.arange(len(d))
ax.barh(y+0.0, d['Macro F1'], height=0.27, color=ACCENT, label='Macro F1')
ax.barh(y-0.28, d['Balanced Accuracy'], height=0.27, color=ACCENT2, label='Balanced Accuracy')
ax.barh(y+0.28, d['Accuracy'], height=0.27, color=GREY, label='Accuracy')
ax.set_yticks(y); ax.set_yticklabels(d['Model'])
ax.set_xlabel('Score'); ax.set_xlim(0,1.0)
ax.set_title('Cell-Type Classification Performance on PBMC 3k\n(5-fold stratified cross-validation)', fontsize=11)
ax.legend(loc='lower right', frameon=False); ax.axvline(0, color='k', lw=0.5)
plt.tight_layout(); plt.savefig('figures/fig1_model_comparison.png', bbox_inches='tight'); plt.close()

# ---- Figure 2: accuracy vs macro F1 gap ----
fig, ax = plt.subplots(figsize=(8,5.5))
order = df.sort_values('Accuracy', ascending=False)
x = np.arange(len(order))
ax.plot(x, order['Accuracy'], 'o-', color=GREY, label='Accuracy', markersize=7)
ax.plot(x, order['Macro F1'], 'o-', color=ACCENT, label='Macro F1', markersize=7)
ax.plot(x, order['Balanced Accuracy'], 'o-', color=ACCENT2, label='Balanced Accuracy', markersize=7)
ax.fill_between(x, order['Macro F1'], order['Accuracy'], alpha=0.1, color=ACCENT2)
ax.set_xticks(x); ax.set_xticklabels(order['Model'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Score'); ax.set_ylim(0,1.0)
ax.set_title('Accuracy Overstates Performance Under Class Imbalance', fontsize=11)
ax.legend(frameon=False)
plt.tight_layout(); plt.savefig('figures/fig2_accuracy_gap.png', bbox_inches='tight'); plt.close()

# ---- Figure 3: per-class F1 heatmap ----
pc_sorted = pc.set_index('Model').loc[df.sort_values('Macro F1',ascending=False)['Model']]
mat = pc_sorted[CLASSES].values
fig, ax = plt.subplots(figsize=(10,5.5))
im = ax.imshow(mat, cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels([f'{c}\n(n={COUNTS[c]})' for c in CLASSES], fontsize=8, rotation=30, ha='right')
ax.set_yticks(range(len(pc_sorted))); ax.set_yticklabels(pc_sorted.index, fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=7,
                color='white' if mat[i,j]<0.5 else 'black')
ax.set_title('Per-Class F1 Score by Model (columns ordered common → rare)', fontsize=11)
plt.colorbar(im, label='F1 score', fraction=0.025)
plt.tight_layout(); plt.savefig('figures/fig3_per_class_heatmap.png', bbox_inches='tight'); plt.close()

# ---- Figure 4: confusion matrix (best model) ----
conf = np.load('results/confusion_best.npy')
conf_norm = conf / conf.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.5,6.5))
im = ax.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
cls = meta['classes']
ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(cls))); ax.set_yticklabels(cls, fontsize=8)
for i in range(len(cls)):
    for j in range(len(cls)):
        if conf_norm[i,j] > 0.01:
            ax.text(j,i,f'{conf_norm[i,j]:.2f}',ha='center',va='center',fontsize=7,
                    color='white' if conf_norm[i,j]>0.5 else 'black')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix — {meta["best_model"]}\n(row-normalized, aggregated over folds)', fontsize=11)
plt.colorbar(im, label='Fraction', fraction=0.045)
plt.tight_layout(); plt.savefig('figures/fig4_confusion.png', bbox_inches='tight'); plt.close()

# ---- Figure 5: per-class F1 vs frequency (honest scatter + trend) ----
fig, ax = plt.subplots(figsize=(8.5,6))
all_x, all_y = [], []
for _, row in pc.iterrows():
    for c in CLASSES:
        ax.scatter(COUNTS[c], row[c], s=18, color=GREY, alpha=0.4, linewidths=0, zorder=1)
        all_x.append(COUNTS[c]); all_y.append(row[c])
mean_f1 = {c: pc[c].mean() for c in CLASSES}
mx = [COUNTS[c] for c in CLASSES]; my = [mean_f1[c] for c in CLASSES]
ax.scatter(mx, my, s=90, color=ACCENT, zorder=3, label='Mean F1 across models',
           edgecolors='white', linewidths=1)
lx = np.log10(mx); coef = np.polyfit(lx, my, 1)
xs = np.linspace(min(lx), max(lx), 100)
ax.plot(10**xs, np.polyval(coef, xs), '--', color=ACCENT2, lw=2, zorder=2,
        label=f'Trend (slope={coef[0]:.2f} F1 per decade)')
for c in CLASSES:
    ax.annotate(c, (COUNTS[c], mean_f1[c]), fontsize=7, color='#334155',
                xytext=(0,8), textcoords='offset points', ha='center')
r = np.corrcoef(np.log10(all_x), all_y)[0,1]
ax.set_xscale('log')
ax.set_xlabel('Number of cells in class (log scale)'); ax.set_ylabel('Per-class F1 score')
ax.set_ylim(-0.03,1.08)
ax.set_title(f'Rarer Cell Types Are Harder to Classify (r = {r:.2f})\n'
             'grey = individual models; blue = per-class mean; trend positive but noisy', fontsize=11)
ax.legend(loc='lower right', frameon=False)
plt.tight_layout(); plt.savefig('figures/fig5_f1_vs_frequency.png', bbox_inches='tight'); plt.close()

# ---- Figure 6: UMAP embedding ----
a = sc.read_h5ad('data/pbmc3k.h5ad')
sc.pp.pca(a, n_comps=50)
sc.pp.neighbors(a, n_neighbors=15, random_state=42)
sc.tl.umap(a, random_state=42)
umap = a.obsm['X_umap']; labels = a.obs['louvain'].values
fig, ax = plt.subplots(figsize=(9,7))
palette = plt.cm.tab10(np.linspace(0,1,len(CLASSES)))
for i, ct in enumerate([c for c in CLASSES if c in set(labels)]):
    m = labels == ct
    ax.scatter(umap[m,0], umap[m,1], s=8, color=palette[i],
               label=f'{ct} (n={COUNTS.get(ct,"?")})', alpha=0.75, linewidths=0)
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
ax.set_title('UMAP Embedding of PBMC 3k, Colored by Annotated Cell Type', fontsize=11)
ax.legend(loc='upper right', frameon=False, fontsize=7, markerscale=1.5); ax.grid(False)
plt.tight_layout(); plt.savefig('figures/fig6_umap.png', bbox_inches='tight'); plt.close()

print("All six figures generated.")
