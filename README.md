# Benchmarking Machine Learning Methods for Single-Cell Cell-Type Classification

A controlled, fully reproducible benchmark of ten machine learning classifiers for automated cell-type annotation in single-cell RNA-sequencing (scRNA-seq) data, with emphasis on class imbalance and rare cell-type recovery.

**Companion article:** https://medium.com/@selma.marrakchi/benchmarking-ml-methods-for-cell-type-classification-in-scrna-seq-8719270c4361

---

## Summary

Automated cell-type classification is a routine step in single-cell analysis, but method choice and evaluation protocol are often treated casually. This benchmark holds preprocessing, feature space, and evaluation fixed while varying only the classifier, on a canonical PBMC dataset (2,638 cells, 8 cell types, 76:1 class imbalance).

**Key findings:**

1. **Class-weighted logistic regression wins** (macro F1 = 0.929), beating a neural network and all tree ensembles — and training in ~1 second.
2. **Accuracy is misleading under imbalance.** The RBF SVM scores 93% accuracy but only 0.75 macro F1; reporting accuracy alone would rank it far too highly.
3. **Class weighting dramatically improves rare-cell-type recovery** — up to 4× F1 improvement for the rarest populations, at negligible cost to common classes.
4. **k-NN collapses** in the 1,838-dimensional expression space (macro F1 = 0.25), failing entirely on three of eight cell types — a clear curse-of-dimensionality result.

---

## Repository Structure

```
scrna-celltype-benchmark/
│
├── data/
│   └── pbmc3k.h5ad              # Dataset (downloaded via download_data.py)
│
├── results/
│   ├── model_comparison.csv     # Aggregate metrics per model
│   ├── per_class_f1.csv         # Per-class F1 for every model
│   ├── confusion_best.npy       # Confusion matrix of best model
│   └── meta.json                # Run metadata
│
├── figures/
│   ├── fig1_model_comparison.png
│   ├── fig2_accuracy_gap.png
│   ├── fig3_per_class_heatmap.png
│   └── fig4_confusion.png
│
├── download_data.py            # Fetch the dataset
├── benchmark.py                # Core benchmarking engine
├── make_figures.py             # Generate all figures from results
├── requirements.txt
└── README.md
```

---

## Dataset

**PBMC 3k** — peripheral blood mononuclear cells from a healthy donor, generated with 10x Genomics droplet-based sequencing. After QC filtering: 2,638 cells × 1,838 highly variable genes, with 8 annotated immune cell types (log-normalized, scaled expression).

| Cell type | Cells | Proportion |
|-----------|-------|-----------|
| CD4 T cells | 1,144 | 43.4% |
| CD14+ Monocytes | 480 | 18.2% |
| B cells | 342 | 13.0% |
| CD8 T cells | 316 | 12.0% |
| NK cells | 154 | 5.8% |
| FCGR3A+ Monocytes | 150 | 5.7% |
| Dendritic cells | 37 | 1.4% |
| Megakaryocytes | 15 | 0.6% |

Cell-type labels derive from the standard Louvain clustering + annotation workflow and are treated as ground truth for the classification task.

---

## Installation

```bash
git clone https://github.com/m-selma/scRNA-seq-celltype-benchmark.git
cd scRNA-seq-celltype-benchmark
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Download the dataset
python download_data.py

# 2. Run the full benchmark (5-fold stratified CV, all 10 models)
python benchmark.py

# 3. Generate figures from the results
python make_figures.py
```

The full benchmark runs in a few minutes on a standard laptop — no GPU required.

---

## Models Evaluated

| Family | Models |
|--------|--------|
| Linear | Logistic regression (±class weighting), linear SVM |
| Kernel | RBF SVM (±class weighting) |
| Tree ensemble | Random forest (±class weighting), histogram gradient boosting |
| Instance-based | k-nearest neighbors (k=15) |
| Neural network | Multilayer perceptron (256, 128) |

---

## Methodology

- **Feature space:** 1,838 highly variable genes, log-normalized and scaled. Identical for all models.
- **Evaluation:** stratified 5-fold cross-validation, fixed seed (42).
- **Metrics:** accuracy, balanced accuracy, macro-averaged F1, weighted F1, plus per-class F1 and confusion matrices.
- **Class weighting:** where supported, loss is weighted inversely to class frequency, isolating the effect of imbalance-aware training.

---

## References

Abdelaal, T., Michielsen, L., Cats, D., Hoogduin, D., Mei, H., Reinders, M. J. T., & Mahfouz, A. (2019). A comparison of automatic cell identification methods for single-cell RNA sequencing data. Genome Biology, 20(1), 194.

Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., & Wang, B. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. Nature Methods, 21(8), 1470–1480.

Hao, M., Gong, J., Zeng, X., Liu, C., Guo, Y., Cheng, X., Wang, T., Ma, J., Zhang, X., & Song, L. (2024). Large-scale foundation model on single-cell transcriptomics. Nature Methods, 21(8), 1481–1491.

Hou, W., & Ji, Z. (2024). Assessing GPT-4 for cell type annotation in single-cell RNA-seq analysis. Nature Methods, 21(8), 1462–1465.

Luecken, M. D., & Theis, F. J. (2019). Current best practices in single-cell RNA-seq analysis: a tutorial. Molecular Systems Biology, 15(6), e8746.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. Genome Biology, 19(1), 15.

Zheng, G. X. Y., et al. (2017). Massively parallel digital transcriptional profiling of single cells. Nature Communications, 8, 14049.

---

## License

MIT License. The PBMC 3k dataset is publicly distributed by 10x Genomics.

---

## Citation

```
Marrakchi, S. (2026). Benchmarking Machine Learning Methods for Automated
Cell-Type Classification in Single-Cell RNA-Sequencing.
GitHub. https://github.com/m-selma/scrna-celltype-benchmark
```
