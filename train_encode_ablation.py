# -*- coding: utf-8 -*-
import os
import sys
import csv
import json
import time
import random
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from attention_blocks import SEBlock1d, ECABlock1d, CBAMBlock1d

_LUT = np.zeros(256, dtype=np.int64)
for _ch, _v in {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}.items():
    _LUT[ord(_ch)] = _v

_BASE_DIM = {"onehot4": 4, "enac_true": 4, "enac_bin": 4, "enac_onehot16": 16,
             "ncp": 3, "ncp4": 4}

_FUSION = {
    "multi_k12":        ("onehot4", "enac_onehot16"),
    "enac_bin_ncp":     ("enac_bin", "ncp4"),
    "onehot16_ncp":     ("enac_onehot16", "ncp4"),
    "onehot4_enac_bin": ("onehot4", "enac_bin"),
}
for _fk, (_fa, _fb) in _FUSION.items():
    _BASE_DIM[_fk] = _BASE_DIM[_fa] + _BASE_DIM[_fb]

N_POS = 2
ENCODING_DIM = dict(_BASE_DIM)
ENCODING_DIM.update({k + "_pos": v + N_POS for k, v in _BASE_DIM.items()})

ENCODING_DESC = {
    "onehot4": "single-base one-hot (4 channels: A/C/G/T, independent and interpretable)",
    "enac_true": "ENAC as used by CLSM6A: window 2, step 1, normalised A/C/G/U counts"
                 " (4 channels; AG and GA map to the same vector, so order is lost)",
    "enac_bin": "binary decomposition of the dinucleotide index (train.py::enac_encode_native)"
                " (4 channels; order kept, each channel is a binary bit with no base meaning)",
    "enac_onehot16": "dinucleotide one-hot (16 channels, one per ordered pair)",
    "ncp": "NCP (3 channels: ring structure / H-bond / amino group)",
    "ncp4": "NCP, 4-channel version (first three as above, fourth is a constant placeholder)",
}
for _fk, (_fa, _fb) in _FUSION.items():
    ENCODING_DESC[_fk] = ("fused encoding: %s (+) %s (%d channels)"
                          % (_fa, _fb, _BASE_DIM[_fk]))

for _k, _d in _BASE_DIM.items():
    ENCODING_DESC[_k + "_pos"] = (
        "%s + positional prior (%d Gaussian position channels appended),"
        "%d channels in total" % (ENCODING_DESC[_k].split(" (")[0], N_POS, _d + N_POS))

_ENAC_TRUE = {
    "AA": [1, 0, 0, 0], "AC": [.5, .5, 0, 0], "AG": [.5, 0, .5, 0], "AU": [.5, 0, 0, .5],
    "CA": [.5, .5, 0, 0], "CC": [0, 1, 0, 0], "CG": [0, .5, .5, 0], "CU": [0, .5, 0, .5],
    "GA": [.5, 0, .5, 0], "GC": [0, .5, .5, 0], "GG": [0, 0, 1, 0], "GU": [0, 0, .5, .5],
    "UA": [.5, 0, 0, .5], "UC": [0, .5, 0, .5], "UG": [0, 0, .5, .5], "UU": [0, 0, 0, 1],
}
_ENAC_TRUE_LUT = np.array([_ENAC_TRUE[k] for k in
                           ["AA", "AC", "AG", "AU", "CA", "CC", "CG", "CU",
                            "GA", "GC", "GG", "GU", "UA", "UC", "UG", "UU"]],
                          dtype=np.float32)

_NCP_TABLE = {
    "A": [1, 0, 1],
    "C": [0, 1, 1],
    "G": [1, 1, 0],
    "T": [0, 0, 0],
}
_NCP_LUT = np.array([_NCP_TABLE[k] for k in "ACGT"], dtype=np.float32)            # (4, 3)
_NCP4_LUT = np.concatenate([_NCP_LUT, np.zeros((4, 1), dtype=np.float32)], axis=1)  # (4, 4)

_POS_CENTER = 100.0
_POS_SIGMAS = (3.0, 10.0)


def _pos_channels(L, center=_POS_CENTER, sigmas=_POS_SIGMAS):
    idx = np.arange(L, dtype=np.float32)
    return np.stack([np.exp(-((idx - center) ** 2) / (2.0 * s * s)) for s in sigmas],
                    axis=1).astype(np.float32)


def _encode_single(kind, base, dinuc, n, L, rows, cols):
    if kind == "onehot4":
        out = np.zeros((n, L, 4), dtype=np.float32)
        out[rows, cols, base[:, :L]] = 1.0
        return out
    if kind == "enac_true":
        return _ENAC_TRUE_LUT[dinuc]
    if kind == "enac_bin":
        shift = np.array([3, 2, 1, 0])[None, None, :]
        return ((dinuc[:, :, None] >> shift) & 1).astype(np.float32)
    if kind == "enac_onehot16":
        out = np.zeros((n, L, 16), dtype=np.float32)
        out[rows, cols, dinuc] = 1.0
        return out
    if kind == "ncp":
        return _NCP_LUT[base[:, :L]]
    if kind == "ncp4":
        return _NCP4_LUT[base[:, :L]]
    raise ValueError("unknown base encoding: %s" % kind)


def encode_sequences(seq_list, kind, seq_len=201):
    use_pos = kind.endswith("_pos")
    base_kind = kind[:-4] if use_pos else kind

    n, L = len(seq_list), seq_len - 1
    base = np.zeros((n, seq_len), dtype=np.int64)
    for i, s in enumerate(seq_list):
        b = np.frombuffer(str(s).strip().upper().encode("ascii", "replace")[:seq_len],
                          dtype=np.uint8)
        base[i, :len(b)] = _LUT[b]

    rows = np.arange(n)[:, None]
    cols = np.arange(L)[None, :]
    dinuc = base[:, :-1] * 4 + base[:, 1:]

    if base_kind in _FUSION:
        a, b = _FUSION[base_kind]
        out = np.concatenate([
            _encode_single(a, base, dinuc, n, L, rows, cols),
            _encode_single(b, base, dinuc, n, L, rows, cols)], axis=2)
    else:
        out = _encode_single(base_kind, base, dinuc, n, L, rows, cols)

    if use_pos:
        pos = _pos_channels(L)
        out = np.concatenate([out, np.broadcast_to(pos[None], (n, L, pos.shape[1]))],
                             axis=2)
    return out


class AblationCNN(nn.Module):

    def __init__(self, in_channels=4, attn="none", reduction=4, eca_k=3,
                 region=None, region_ch=32):
        super().__init__()
        self.attn = attn
        self.region = region
        self.conv1 = self._conv_block(in_channels, 64)
        self.conv2 = self._conv_block(64, 64)
        self.conv3 = self._conv_block(64, 64)

        if attn == "se":
            self.a1 = SEBlock1d(64, reduction)
            self.a2 = SEBlock1d(64, reduction)
            self.a3 = SEBlock1d(64, reduction)
        elif attn == "eca":
            self.a1 = ECABlock1d(64, eca_k)
            self.a2 = ECABlock1d(64, eca_k)
            self.a3 = ECABlock1d(64, eca_k)
        elif attn == "cbam":
            self.a1 = CBAMBlock1d(64, reduction)

        if region is not None:
            lo, hi = region
            self.local = nn.Sequential(
                nn.Conv1d(in_channels, region_ch, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(region_ch, region_ch, kernel_size=5, padding=2),
                nn.ReLU())
            self.local_dim = region_ch * (hi - lo)
        else:
            self.local = None
            self.local_dim = 0

        self.fc = nn.Sequential(nn.Linear(1600 + self.local_dim, 256),
                                nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    @staticmethod
    def _conv_block(i, o):
        return nn.Sequential(
            nn.Conv1d(in_channels=i, out_channels=o, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2))

    def forward(self, x):                       # x: [B, 200, C]
        z = x.transpose(1, 2)                   # -> [B, C, 200]
        local_feat = None
        if self.local is not None:
            lo, hi = self.region
            local_feat = self.local(z[:, :, lo:hi]).flatten(1)

        x = self.conv1(z)
        if self.attn in ("se", "eca"):
            x = self.a1(x)
        x = self.conv2(x)
        if self.attn in ("se", "eca"):
            x = self.a2(x)
        x = self.conv3(x)
        if self.attn in ("se", "eca"):
            x = self.a3(x)
        elif self.attn == "cbam":
            x = self.a1(x)
        g = x.view(x.size(0), -1)
        if local_feat is not None:
            g = torch.cat([g, local_feat], dim=1)
        return self.classifier(self.fc(g))


ATTN_CHANNELS = 64
ATTN_REDUCTION = 4


def attn_bottleneck(channels=ATTN_CHANNELS, reduction=ATTN_REDUCTION):
    return max(1, channels // reduction)


MODEL_ATTN = {"cnn": "none", "cnn_se": "se", "cnn_eca": "eca", "cnn_cbam": "cbam",
              "cnn_region": "none", "cnn_region_ctrl": "none"}

REGION_CORE = (92, 108)
REGION_CTRL = (0, 16)
MODEL_REGION = {"cnn_region": REGION_CORE, "cnn_region_ctrl": REGION_CTRL}


class ArrayDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor([self.y[i]], dtype=torch.float32)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / den if den > 0 else 0.0
    return {"AUC": float(roc_auc_score(y_true, y_prob)),
            "AP": float(average_precision_score(y_true, y_prob)),
            "MCC": float(mcc), "Acc": float(acc),
            "Sen": float(sen), "Spe": float(spe)}


def run_one(kind, model_name, seed, args, device, tr_x, tr_y, va_x, va_y, te_x, te_y, log):
    set_seed(seed)
    tr_ld = DataLoader(ArrayDataset(tr_x, tr_y), batch_size=args.batch_size,
                       shuffle=True, num_workers=0)
    va_ld = DataLoader(ArrayDataset(va_x, va_y), batch_size=args.batch_size * 2)
    te_ld = DataLoader(ArrayDataset(te_x, te_y), batch_size=args.batch_size * 2)

    attn = MODEL_ATTN[model_name]
    in_ch = ENCODING_DIM[kind]
    model = AblationCNN(in_channels=in_ch, attn=attn,
                        region=MODEL_REGION.get(model_name)).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_auc, best_state, best_ep, bad = -1.0, None, -1, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            loss = criterion(model(bx), by)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        vt, vp = [], []
        with torch.no_grad():
            for bx, by in va_ld:
                vp.extend(model(bx.to(device)).cpu().numpy().ravel().tolist())
                vt.extend(by.numpy().ravel().tolist())
        auc = roc_auc_score(vt, vp)
        if auc > best_auc:
            best_auc, best_ep, bad = auc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    tt, tp = [], []
    with torch.no_grad():
        for bx, by in te_ld:
            tp.extend(model(bx.to(device)).cpu().numpy().ravel().tolist())
            tt.extend(by.numpy().ravel().tolist())
    m = calc_metrics(tt, tp)
    _state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {"encoding": kind, "model": model_name, "seed": seed, "in_channels": in_ch,
            "attn_channels": ATTN_CHANNELS, "attn_bottleneck": attn_bottleneck(),
            "params": n_param, "best_epoch": best_ep, "val_auc": best_auc,
            "test_probs": tp, "test_labels": tt, "state": _state, **m}


def plot_summary(df, out_png, models, cell=""):
    encs = list(dict.fromkeys(df["encoding"]))
    base = "cnn"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    n_m = len(models)
    w = 0.8 / n_m
    x = np.arange(len(encs))
    colors = {"cnn": "#7F8C8D", "cnn_se": "#E67E22", "cnn_eca": "#2E86C1",
              "cnn_cbam": "#8E44AD"}
    for i, m in enumerate(models):
        vals, errs = [], []
        for e in encs:
            sub = df[(df.encoding == e) & (df.model == m)]["AUC"]
            vals.append(sub.mean() if len(sub) else np.nan)
            errs.append(sub.std(ddof=0) if len(sub) > 1 else 0.0)
        ax.bar(x + (i - (n_m - 1) / 2) * w, vals, w, yerr=errs, capsize=3,
               color=colors.get(m, "#333333"), label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", "\n") for e in encs], fontsize=9)
    ax.set_ylabel("Test AUC", fontsize=10)
    _pts = []
    for m in models:
        for e in encs:
            s = df[(df.encoding == e) & (df.model == m)]["AUC"]
            if len(s):
                _pts.append((float(s.mean()),
                             float(s.std(ddof=0)) if len(s) > 1 else 0.0))
    _lo = min(a - b for a, b in _pts)
    _hi = max(a + b for a, b in _pts)
    _span = max(_hi - _lo, 0.02)
    ax.set_ylim(_lo - 0.25 * _span, _hi + 0.60 * _span)
    ax.set_title("(a) Test AUC for each encoding x model", fontsize=11)
    for i, m in enumerate(models):
        for j, e in enumerate(encs):
            sub = df[(df.encoding == e) & (df.model == m)]["AUC"]
            if len(sub):
                xx = j + (i - (n_m - 1) / 2) * w
                ax.text(xx, sub.mean() + 0.03 * _span, "%.4f" % sub.mean(),
                        ha="center", fontsize=6.6, rotation=90)
    ax.legend(fontsize=8.5, ncol=2, loc="upper left")
    ax.grid(axis="y", ls=":", alpha=0.45); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    attn_models = [m for m in models if m != base]
    if not attn_models:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "plain CNN only in this run\nno attention module to compare",
                ha="center", va="center", fontsize=12, color="#888888",
                transform=ax.transAxes)
        ax.axhline(0, color="black", lw=0)
    w2 = 0.8 / max(len(attn_models), 1)
    for i, m in enumerate(attn_models):
        deltas, errs = [], []
        for e in encs:
            aa = df[(df.encoding == e) & (df.model == base)]
            bb = df[(df.encoding == e) & (df.model == m)]
            if len(aa) == 0 or len(bb) == 0:
                deltas.append(np.nan)
                errs.append(0.0)
                continue
            deltas.append(float(bb["AUC"].mean() - aa["AUC"].mean()))
            if "fold" in df.columns and len(aa) > 1 and len(bb) > 1:
                mg = aa[["fold", "AUC"]].merge(bb[["fold", "AUC"]], on="fold",
                                               suffixes=("_a", "_b"))
                errs.append(float((mg["AUC_b"] - mg["AUC_a"]).std(ddof=1))
                            if len(mg) > 1 else 0.0)
            elif len(aa) > 1 and len(bb) > 1:
                k = min(len(aa), len(bb))
                errs.append(float((bb["AUC"].values[:k]
                                   - aa["AUC"].values[:k]).std(ddof=1))
                            if k > 1 else 0.0)
            else:
                errs.append(0.0)
        ax.bar(x + (i - (len(attn_models) - 1) / 2) * w2, deltas, w2, yerr=errs,
               capsize=3, color=colors.get(m, "#333333"), label="%s − cnn" % m)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", "\n") for e in encs], fontsize=9)
    ax.set_ylabel("dAUC (attention - plain CNN)", fontsize=10)
    ax.set_title("(b) Does the attention gain depend on the encoding?", fontsize=11)
    if attn_models:
        ax.legend(fontsize=8.5)
    ax.grid(axis="y", ls=":", alpha=0.45); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Encoding x attention ablation (m6A %s)"
                 % (cell or "—"), fontsize=12.5, y=1.02)
    _n_fold = int(df["fold"].nunique()) if "fold" in df.columns else 1
    _n_arm = int(df.groupby(["encoding", "model"])["AUC"].size().max())
    if _n_fold > 1:
        _proto = "%d-fold cross-validation (one seed per fold)" % _n_fold
        _eb = "SD across folds"
    else:
        _proto = "%d independent repeats" % _n_arm
        _eb = "SD across seeds"
    fig.text(0.5, -0.06,
             "%s, %d runs per arm. Left panel error bars: %s;"
             "right panel error bars: SD of per-fold paired differences"
             " (both arms share the same folds, i.e. a paired design)."
             % (_proto, _n_arm, _eb),
             ha="center", fontsize=7.5, color="#555555")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Encoding x attention ablation")
    ap.add_argument("--cell", default="liver")
    ap.add_argument("--data_dir", default=str(BASE / "data" / "original"))
    ap.add_argument("--out_dir", default=str(BASE / "output_encode_ablation"))
    ap.add_argument("--encodings",
                    default="onehot4,enac_true,enac_bin,enac_onehot16,ncp,ncp4")
    ap.add_argument("--models", default="cnn")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--fold_id", type=int, default=-1,
                    help="-1: all folds; k: train only fold k (seed 42+k)")
    ap.add_argument("--folds", type=int, default=5,
                    help=">1: stratified CV inside the training split, every fold scored on the same test set")
    ap.add_argument("--split", default="random", choices=["random", "chrom"],
                    help="random: keep the original split; chrom: hold out whole chromosomes"
                         "to rule out inflation caused by overlapping windows")
    ap.add_argument("--chrom_holdout", default="",
                    help="chromosomes to hold out in chrom mode (comma separated)")
    ap.add_argument("--chrom_ratio", type=float, default=0.2,
                    help="target fraction of samples held out (default 0.2)")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--plot_only", action="store_true",
                    help="skip training; re-aggregate and replot from results.csv")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            try:
                cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
                arch = torch.cuda.get_arch_list()
                if cap not in arch:
                    print("[warn] GPU %s not in torch arch list %s, using CPU"
                          % (cap, arch))
                    device = torch.device("cpu")
            except Exception:
                pass
    else:
        device = torch.device(args.device)
    print("[device] %s" % device)

    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"
    log_fp = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()


    if args.plot_only:
        df = pd.read_csv(out_dir / "results.csv")
        encodings = list(dict.fromkeys(df["encoding"]))
        models = list(dict.fromkeys(df["model"]))
        summary = (df.groupby(["encoding", "model"], sort=False)
                     .agg(AUC_mean=("AUC", "mean"),
                          AUC_std=("AUC", lambda s: s.std(ddof=0)),
                          AP_mean=("AP", "mean"),
                          MCC_mean=("MCC", "mean"),
                          Acc_mean=("Acc", "mean"),
                          n_seed=("AUC", "size"))
                     .reset_index())
        summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
        _KNOWN = ["HEK293T", "HEK293", "MOLM13", "HCT116", "HepG2", "A549",
                  "CD8T", "HeLa", "liver", "kidney", "brain"]
        cell_for_plot = args.cell
        for _c in _KNOWN:
            if _c.lower() in out_dir.name.lower():
                cell_for_plot = _c
                break
        png = BASE / "pic" / ("fig_encoding_attention_%s.png" % cell_for_plot)
        png.parent.mkdir(parents=True, exist_ok=True)
        plot_summary(df, str(png), models, cell=cell_for_plot)
        for e in encodings:
            a = df[(df.encoding == e) & (df.model == "cnn")]["AUC"]
            for m in models:
                if m == "cnn":
                    continue
                b = df[(df.encoding == e) & (df.model == m)]["AUC"]
        log_fp.close()
        return
    n_arm = args.folds if (args.folds and args.folds > 1) else len(seeds)

    df_tr = pd.read_csv(Path(args.data_dir) / ("%s_train.csv" % args.cell))
    df_te = pd.read_csv(Path(args.data_dir) / ("%s_test.csv" % args.cell))
    tr_seq = df_tr["seq"].astype(str).tolist(); tr_lab = df_tr["label"].astype(int).values
    te_seq = df_te["seq"].astype(str).tolist(); te_lab = df_te["label"].astype(int).values

    if args.split == "chrom":
        if "chr" not in df_tr.columns:
            raise SystemExit("chr column missing; cannot do chromosome hold-out")
        df_all = pd.concat([df_tr, df_te], ignore_index=True)
        chrs = list(pd.unique(df_all["chr"].astype(str)))

        def _ckey(c):
            digits = "".join(ch for ch in str(c) if ch.isdigit())
            return (0, int(digits), str(c)) if digits else (1, 0, str(c))

        chrs = sorted(chrs, key=_ckey)
        if args.chrom_holdout:
            hold = [c.strip() for c in args.chrom_holdout.split(",") if c.strip()]
        else:
            cnt = df_all["chr"].astype(str).value_counts()
            order = [c for c in cnt.index
                     if c in chrs and c.replace("chr", "").isdigit()]
            hold, acc = [], 0
            for c in order:
                if acc / len(df_all) >= args.chrom_ratio:
                    break
                hold.append(c); acc += int(cnt[c])
            if not hold:
                hold = [order[0]]
        mask = df_all["chr"].astype(str).isin(hold).values
        df_tr = df_all[~mask].reset_index(drop=True)
        df_te = df_all[mask].reset_index(drop=True)
        tr_seq = df_tr["seq"].astype(str).tolist()
        tr_lab = df_tr["label"].astype(int).values
        te_seq = df_te["seq"].astype(str).tolist()
        te_lab = df_te["label"].astype(int).values

    if args.folds and args.folds > 1:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        splits = [(tr_i, va_i) for tr_i, va_i in skf.split(np.zeros(len(tr_lab)), tr_lab)]
    else:
        tr_idx, va_idx = train_test_split(np.arange(len(tr_lab)), test_size=args.val_ratio,
                                          random_state=42, stratify=tr_lab)
        splits = [(tr_idx, va_idx)]

    X = {}
    for e in encodings:
        t0 = time.time()
        X[e] = (encode_sequences(tr_seq, e), encode_sequences(te_seq, e))

    rows = []
    t_start = time.time()
    for e in encodings:
        tr_x, te_x = X[e]
        for m in models:
            for fi, (tr_i, va_i) in enumerate(splits):
                if getattr(args, "fold_id", -1) >= 0 and fi != args.fold_id:
                    continue
                sd_list = [seeds[0] + fi] if len(splits) > 1 else seeds
                for sd in sd_list:
                    r = run_one(e, m, sd, args, device,
                                tr_x[tr_i], tr_lab[tr_i],
                                tr_x[va_i], tr_lab[va_i],
                                te_x, te_lab, log)
                    r["fold"] = fi
                    rows.append({k: v for k, v in r.items() if not k.startswith("test_")})
                    np.savez(out_dir / ("pred_%s_%s_seed%d.npz" % (e, m, sd)),
                             y_true=np.array(r["test_labels"]), y_prob=np.array(r["test_probs"]))
                    if r.get("state") is not None:
                        torch.save(r["state"],
                                   out_dir / ("model_%s_%s_seed%d.pkl" % (e, m, sd)))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for e in encodings:
        for m in models:
            sub = df[(df.encoding == e) & (df.model == m)]
            rec = {"encoding": e, "model": m,
                   "AUC_mean": sub.AUC.mean(), "AUC_std": sub.AUC.std(ddof=0),
                   "AP_mean": sub.AP.mean(), "MCC_mean": sub.MCC.mean(),
                   "Acc_mean": sub.Acc.mean(), "n_seed": len(sub)}
            summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    for e in encodings:
        a = df[(df.encoding == e) & (df.model == "cnn")].AUC
        for m in models:
            if m == "cnn":
                continue
            b = df[(df.encoding == e) & (df.model == m)].AUC

    if not args.no_plot and len(encodings) * len(models) >= 2:
        png = BASE / "pic" / ("fig_encoding_attention_%s.png" % args.cell)
        png.parent.mkdir(parents=True, exist_ok=True)
        plot_summary(df, str(png), models, cell=args.cell)

    log_fp.close()


if __name__ == "__main__":
    main()
    print("[done] see results.csv / summary.csv under --out_dir "
          "(default: output_encode_ablation/)")
