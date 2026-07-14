"""
benchmark.py — Core benchmarking engine for single-cell cell-type classification.

Evaluates multiple ML classifiers on the PBMC 3k dataset using stratified
cross-validation, reporting per-class and aggregate metrics with explicit
attention to class imbalance and rare cell-type performance.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import json, time

RANDOM_STATE = 42
N_FOLDS = 5

def load_data():
    a = sc.read_h5ad('data/pbmc3k.h5ad')
    X = a.X.toarray() if hasattr(a.X, 'toarray') else np.asarray(a.X)
    y = a.obs['louvain'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X.astype(np.float32), y_enc, le

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=2000, C=1.0,
                                    random_state=RANDOM_STATE),
        'Logistic Regression (balanced)': LogisticRegression(max_iter=2000, C=1.0,
                                    class_weight='balanced',
                                    random_state=RANDOM_STATE),
        'Linear SVM': SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
        'RBF SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
        'RBF SVM (balanced)': SVC(kernel='rbf', C=1.0, gamma='scale',
                                    class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                    random_state=RANDOM_STATE),
        'Random Forest (balanced)': RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                    class_weight='balanced_subsample',
                                    random_state=RANDOM_STATE),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=150, random_state=RANDOM_STATE),
        'k-NN (k=15)': KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        'MLP (neural net)': MLPClassifier(hidden_layer_sizes=(256,128),
                                    max_iter=300, early_stopping=True,
                                    random_state=RANDOM_STATE),
    }

def run_benchmark():
    X, y, le = load_data()
    print(f"Data: {X.shape[0]} cells x {X.shape[1]} genes, {len(le.classes_)} classes")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()

    rows = []
    per_class_f1 = {name: {c: [] for c in le.classes_} for name in models}
    all_conf = {name: np.zeros((len(le.classes_), len(le.classes_))) for name in models}

    for name, model in models.items():
        t0 = time.time()
        accs, baccs, macro_f1s, weighted_f1s = [], [], [], []
        for tr, te in skf.split(X, y):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            accs.append(accuracy_score(y[te], pred))
            baccs.append(balanced_accuracy_score(y[te], pred))
            macro_f1s.append(f1_score(y[te], pred, average='macro'))
            weighted_f1s.append(f1_score(y[te], pred, average='weighted'))
            # per-class F1
            pcf = f1_score(y[te], pred, average=None, labels=range(len(le.classes_)))
            for i, c in enumerate(le.classes_):
                per_class_f1[name][c].append(pcf[i])
            all_conf[name] += confusion_matrix(y[te], pred, labels=range(len(le.classes_)))
        elapsed = time.time() - t0
        rows.append({
            'Model': name,
            'Accuracy': np.mean(accs),
            'Accuracy_std': np.std(accs),
            'Balanced Accuracy': np.mean(baccs),
            'Balanced Accuracy_std': np.std(baccs),
            'Macro F1': np.mean(macro_f1s),
            'Macro F1_std': np.std(macro_f1s),
            'Weighted F1': np.mean(weighted_f1s),
            'Time (s)': elapsed,
        })
        print(f"  {name:32s} acc={np.mean(accs):.3f}  bal_acc={np.mean(baccs):.3f}  "
              f"macroF1={np.mean(macro_f1s):.3f}  ({elapsed:.1f}s)", flush=True)
        pd.DataFrame(rows).to_csv('results/_partial.csv', index=False)

    df = pd.DataFrame(rows).sort_values('Macro F1', ascending=False).reset_index(drop=True)
    df.to_csv('results/model_comparison.csv', index=False)

    # Per-class F1 (averaged over folds)
    pcf_rows = []
    for name in models:
        row = {'Model': name}
        for c in le.classes_:
            row[c] = np.mean(per_class_f1[name][c])
        pcf_rows.append(row)
    pcf_df = pd.DataFrame(pcf_rows)
    pcf_df.to_csv('results/per_class_f1.csv', index=False)

    # Save confusion matrix for best model
    best = df.iloc[0]['Model']
    np.save('results/confusion_best.npy', all_conf[best])
    with open('results/meta.json','w') as f:
        json.dump({'classes': list(le.classes_), 'best_model': best,
                   'n_cells': int(X.shape[0]), 'n_genes': int(X.shape[1])}, f, indent=2)
    print(f"\nBest model by Macro F1: {best}")
    return df, pcf_df

if __name__ == '__main__':
    run_benchmark()
