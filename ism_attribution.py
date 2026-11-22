# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_spec = importlib.util.spec_from_file_location(
    "tea", str(BASE / "train_encode_ablation.py"))
TEA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TEA)

LSEQ = 200
SITE = 100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def base_channels(enc):
    if enc.endswith("_pos"):
        return TEA.ENCODING_DIM[enc] - TEA.N_POS
    return TEA.ENCODING_DIM[enc]


def train_once(enc, model_name, args, device, log):
    df_tr = pd.read_csv(Path(args.data_dir) / ("%s_train.csv" % args.cell))
    df_te = pd.read_csv(Path(args.data_dir) / ("%s_test.csv" % args.cell))
    tr_seq = df_tr["seq"].astype(str).tolist()
    tr_lab = df_tr["label"].astype(int).values
    te_seq = df_te["seq"].astype(str).tolist()
    te_lab = df_te["label"].astype(int).values

    X_tr = TEA.encode_sequences(tr_seq, enc)
    X_te = TEA.encode_sequences(te_seq, enc)

    tr_idx, va_idx = train_test_split(np.arange(len(tr_lab)), test_size=0.1,
                                      random_state=42, stratify=tr_lab)
    set_seed(args.seed)
    model = TEA.AblationCNN(in_channels=TEA.ENCODING_DIM[enc],
                            attn={"cnn": "none", "cnn_se": "se",
                                  "cnn_eca": "eca", "cnn_cbam": "cbam"}[model_name])
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCELoss()

    def loader(X, y, idx, shuffle):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx]).float()),
            batch_size=args.batch_size, shuffle=shuffle)

    tr_ld = loader(X_tr, tr_lab, tr_idx, True)
    va_x = torch.from_numpy(X_tr[va_idx]).float().to(device)
    va_y = tr_lab[va_idx]

    best_auc, best_state, patience = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device).view(-1, 1)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(va_x).cpu().numpy().ravel()
        auc = roc_auc_score(va_y, pv)
        if auc > best_auc:
            best_auc, patience = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        Xte_t = torch.from_numpy(X_te).float().to(device)
        pte = np.concatenate([model(Xte_t[k:k + 256]).cpu().numpy().ravel()
                              for k in range(0, len(Xte_t), 256)])
    test_auc = roc_auc_score(te_lab, pte)
    return model, te_seq, te_lab, test_auc


def forward_probs(model, X, device, batch=512):
    Xt = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        return np.concatenate([model(Xt[k:k + batch]).cpu().numpy().ravel()
                               for k in range(0, len(Xt), batch)])


def ism_scan(model, seqs, enc, device, batch=512, log=None):
    N, L = len(seqs), LSEQ
    alpha = "ACGU" if any("U" in str(s) for s in seqs[:64]) else "ACGT"

    base_p = forward_probs(model, TEA.encode_sequences(seqs, enc), device, batch)

    imp = np.zeros((N, L, 4), dtype=np.float32)
    t0 = time.time()
    for pos in range(L):
        variants, owners = [], []
        for b, ch in enumerate(alpha):
            for i, s in enumerate(seqs):
                s = str(s)
                if pos < len(s) and s[pos] != ch:
                    variants.append(s[:pos] + ch + s[pos + 1:])
                    owners.append((i, b))
        if not variants:
            continue
        pm = forward_probs(model, TEA.encode_sequences(variants, enc), device, batch)
        for (i, b), p in zip(owners, pm):
            imp[i, pos, b] = base_p[i] - p
    return imp


def position_profile(imp):
    return imp.max(axis=2).mean(axis=0)


def base_freq_diff(cell, data_dir=None):
    df = pd.read_csv(Path(data_dir or (BASE / "data" / "original")) / ("%s_train.csv" % cell))
    seqs = df["seq"].astype(str).tolist()
    lab = df["label"].astype(int).values
    lut = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    arr = np.full((len(seqs), LSEQ), -1, dtype=np.int8)
    for i, s in enumerate(seqs):
        s = s.strip().upper()[:LSEQ]
        for j, ch in enumerate(s):
            if ch in lut:
                arr[i, j] = lut[ch]
    pos, neg = arr[lab == 1], arr[lab == 0]
    fp = np.stack([(pos == b).mean(axis=0) for b in range(4)], axis=1)
    fn = np.stack([(neg == b).mean(axis=0) for b in range(4)], axis=1)
    return fp - fn


COLORS = {"onehot4": "#7F8C8D", "enac_true": "#27AE60",
          "enac_bin": "#E67E22", "enac_onehot16": "#8E44AD",
          "ncp": "#2E86C1", "ncp4": "#5DADE2"}


def plot_profiles(profiles, freqs, out_png, cell, lo=90, hi=112):
    n = len(profiles)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = axes[0]
    for enc, pr in profiles.items():
        ax.plot(np.arange(LSEQ), pr, lw=1.6,
                color=COLORS.get(enc, None), label=enc)
    ax.axvline(SITE, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("Sequence position (0-based)", fontsize=10.5)
    ax.set_ylabel("ISM importance", fontsize=10.5)
    ax.set_title("(a) Per-position contribution (mean over the test set)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(ls=":", alpha=0.45); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    for enc, pr in profiles.items():
        ax.plot(np.arange(lo, hi), pr[lo:hi], lw=2.0, marker="o", ms=3,
                color=COLORS.get(enc, None), label=enc)
    if freqs is not None:
        ax2 = ax.twinx()
        diff = np.abs(freqs[lo:hi]).max(axis=1)
        ax2.bar(np.arange(lo, hi), diff, width=0.62, color="#BDC3C7",
                alpha=0.45, zorder=0, label="Base frequency difference (pos - neg)")
        ax2.set_ylabel("Base frequency difference (pos - neg)", fontsize=9.5, color="#7F8C8D")
        ax2.tick_params(axis="y", colors="#7F8C8D", labelsize=9)
        ax2.legend(fontsize=8, loc="upper right")
    ax.axvline(SITE, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("Sequence position", fontsize=10.5)
    ax.set_ylabel("ISM importance", fontsize=10.5)
    ax.set_title("(b) Zoom on the candidate site", fontsize=11)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top",):
        ax.spines[s].set_visible(False)

    fig.suptitle("Position-level attribution (m6A %s)" % cell,
                 fontsize=12.5, y=1.02)
    fig.text(0.5, -0.045,
             "ISM replaces each position with the other three bases and records the drop in predicted probability,"
             "taking the maximum over the four substitutions; grey bars are the base-frequency difference.",
             ha="center", fontsize=7.8, color="#555555")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="ISM position-level attribution")
    ap.add_argument("--cell", default="liver")
    ap.add_argument("--encodings", default="onehot4,enac_true,enac_bin,enac_onehot16")
    ap.add_argument("--models", default="cnn")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default=str(BASE / "output_ism"))
    ap.add_argument("--data_dir", default=str(BASE / "data" / "original"),
                    help="data directory; point to new_data for the rebuilt datasets")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = open(out_dir / "ism_log.txt", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n"); log_fp.flush()

    encs = [e.strip() for e in args.encodings.split(",") if e.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    device = torch.device(args.device)


    freqs = base_freq_diff(args.cell, args.data_dir)

    profiles = {}
    rows = []
    for enc in encs:
        for m in models:
            t0 = time.time()
            model, te_seqs, te_lab, auc = train_once(enc, m, args, device, log)
            imp = ism_scan(model, te_seqs, enc, device, log=log)
            prof = position_profile(imp)
            np.savez(out_dir / ("ism_%s_%s.npz" % (enc, m)),
                     importance=imp, profile=prof, y_true=te_lab, test_auc=auc)
            profiles["%s" % enc] = prof
            rows.append(dict(encoding=enc, model=m, test_auc=auc,
                             importance_first100=float(prof[:100].mean()),
                             importance_site=float(prof[98:103].mean()),
                             site_ratio=float(prof[98:103].sum() / prof.sum()),
                             peak_position=int(np.argmax(prof)),
                             minutes=(time.time() - t0) / 60))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ism_summary.csv", index=False, encoding="utf-8-sig")

    if len(models) == 1:
        png = out_dir / ("ism_%s_%s.png" % (args.cell, models[0]))
        plot_profiles(profiles, freqs, png, args.cell)
    log_fp.close()


if __name__ == "__main__":
    main()
