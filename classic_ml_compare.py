# -*- coding: utf-8 -*-
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, accuracy_score, confusion_matrix)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "original"

from train_encode_ablation import encode_sequences

DATASETS = ["liver", "kidney", "brain", "HCT116", "HEK293"]
ENCODINGS = ["onehot4", "enac_true", "enac_bin", "enac_onehot16", "ncp", "ncp4"]


def make_model(tag, seed, n_jobs):
    if tag == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=n_jobs,
                                      random_state=seed, min_samples_leaf=1)
    if tag == "ada":
        return AdaBoostClassifier(n_estimators=50, random_state=seed)
    if tag == "knn":
        return KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=n_jobs)
    if tag == "hgb":
        return HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    if tag == "lr":
        return LogisticRegression(max_iter=1500, solver="lbfgs", random_state=seed)
    raise ValueError("unknown algorithm: %s" % tag)


def metrics(y, p):
    y = np.asarray(y).ravel().astype(int)
    p = np.asarray(p).ravel()
    pred = (p >= 0.5).astype(int) if p.min() >= 0 and p.max() <= 1 else (p >= 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, p), AP=average_precision_score(y, p),
                MCC=matthews_corrcoef(y, pred), Acc=accuracy_score(y, pred),
                Sen=tp / (tp + fn) if (tp + fn) else 0.0,
                Spe=tn / (tn + fp) if (tn + fp) else 0.0)


def load_cell(cell):
    tr = pd.read_csv(DATA / ("%s_train.csv" % cell))
    te = pd.read_csv(DATA / ("%s_test.csv" % cell))
    return (tr["seq"].astype(str).tolist(), tr["label"].astype(int).values,
            te["seq"].astype(str).tolist(), te["label"].astype(int).values)


def main():
    ap = argparse.ArgumentParser(description="Classical ML baselines")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--encodings", default=",".join(ENCODINGS))
    ap.add_argument("--models", default="rf,ada,knn,hgb,lr")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=6, help="number of parallel jobs")
    ap.add_argument("--pca", type=int, default=0,
                    help="reduce features to this dimension (0 = no reduction)."
                         "separates the effect of information content from that of feature dimension:"
                         "encodings differ 4x in dimensionality and classical ML is sensitive to it.")
    ap.add_argument("--proj", default="pca", choices=["pca", "random"],
                    help="reduction method: pca = top-k components by variance;"
                         "random = Gaussian random projection."
                         "PCA ranks by variance, while the discriminative directions of sequence encodings are often low-variance"
                         "(only one or two of the 16 channels matter at the key positions),"
                         "so PCA discards the discriminative signal;"
                         "use random to isolate the dimension effect.")
    ap.add_argument("--out_dir", default=str(BASE / "output_classic_ml"))
    ap.add_argument("--data_dir", default=str(BASE / "data" / "original"),
                    help="directory holding {cell}_train.csv / {cell}_test.csv")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    global DATA
    DATA = Path(args.data_dir)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cfg = len(datasets) * len(encodings) * len(models) * args.folds
    print("=" * 92)
    print("Classical ML comparison | %d datasets x %d encodings x %d algorithms x %d folds = %d fits"
          % (len(datasets), len(encodings), len(models), args.folds, n_cfg))
    print("=" * 92)
    if args.list:
        return

    log_fp = open(out_dir / "classic_ml_log.txt", "a", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        log_fp.write(m + "\n")
        log_fp.flush()

    t_all = time.time()
    for cell in datasets:
        out_csv = out_dir / ("results_%s.csv" % cell)
        if out_csv.is_file():
            continue
        X_tr_s, y_tr, X_te_s, y_te = load_cell(cell)
        rows = []
        t_ds = time.time()
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        for enc in encodings:
            t0 = time.time()
            Xtr = encode_sequences(X_tr_s, enc)
            Xte = encode_sequences(X_te_s, enc)
            Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float32)
            Xte = Xte.reshape(len(Xte), -1).astype(np.float32)
            for mi, mtag in enumerate(models):
                aucs = []
                for fi, (tr_idx, _) in enumerate(skf.split(Xtr, y_tr)):
                    sc = StandardScaler().fit(Xtr[tr_idx])
                    Xa, Xb = sc.transform(Xtr[tr_idx]), sc.transform(Xte)
                    if args.pca and args.pca < Xa.shape[1]:
                        if args.proj == "random":
                            from sklearn.random_projection import GaussianRandomProjection
                            dr = GaussianRandomProjection(n_components=args.pca,
                                                          random_state=42).fit(Xa)
                        else:
                            from sklearn.decomposition import PCA
                            dr = PCA(n_components=args.pca, random_state=42).fit(Xa)
                        Xa, Xb = dr.transform(Xa), dr.transform(Xb)
                    clf = make_model(mtag, 42 + fi, args.jobs)
                    clf.fit(Xa, y_tr[tr_idx])
                    if hasattr(clf, "predict_proba"):
                        p = clf.predict_proba(Xb)[:, 1]
                    else:
                        p = clf.decision_function(Xb)
                    m = metrics(y_te, p)
                    aucs.append(m["AUC"])
                    rows.append(dict(dataset=cell, encoding=enc, model=mtag,
                                     fold=fi, seed=42 + fi, pca=args.pca, **m))
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        summ = (df.groupby(["encoding", "model"], sort=False)
                  .agg(AUC_mean=("AUC", "mean"),
                       AUC_std=("AUC", lambda s: s.std(ddof=0)),
                       AP_mean=("AP", "mean"), MCC_mean=("MCC", "mean"),
                       Acc_mean=("Acc", "mean"), n=("AUC", "size"))
                  .reset_index())
        summ.to_csv(out_dir / ("summary_%s.csv" % cell), index=False, encoding="utf-8-sig")

    log_fp.close()


if __name__ == "__main__":
    main()
