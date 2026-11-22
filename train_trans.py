# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "original"

_LUT = np.zeros(256, dtype=np.int8)
for _ch, _v in {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}.items():
    _LUT[ord(_ch)] = _v


def encode_sequences(seq_list, seq_len=201):
    out = np.zeros((len(seq_list), seq_len), dtype=np.int8)
    for i, s in enumerate(seq_list):
        s = str(s).strip().upper()
        if len(s) < seq_len:
            pad = seq_len - len(s)
            left = pad // 2
            s = "A" * left + s + "A" * (pad - left)
        b = s[:seq_len].encode("ascii", "replace")
        if len(b) < seq_len:
            b = b + b"A" * (seq_len - len(b))
        out[i] = _LUT[np.frombuffer(b, dtype=np.uint8)]
    return out


class RNAm6ADataset(Dataset):

    def __init__(self, encoded, labels, augment_shift=0):
        self.x = encoded
        self.y = np.asarray(labels, dtype=np.float32)
        self.augment_shift = augment_shift

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment_shift > 0:
            x = x.copy()
            n = x.shape[0]
            c = n // 2
            mask = np.random.rand(n) < 0.05
            mask[max(0, c - 10):c + 11] = False
            x[mask] = np.random.randint(0, 4, size=int(mask.sum())).astype(np.int8)
        return torch.from_numpy(x).long(), torch.tensor([self.y[idx]], dtype=torch.float32)


class RNAFormerEncoder(nn.Module):

    def __init__(self, seq_len=201, kmer=4, stride=1, d_model=96, nhead=4,
                 layers=4, dim_ff=192, dropout=0.15, pos_mode="abs",
                 center_offset=-1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.seq_len = seq_len
        self.kmer = kmer
        self.stride = stride
        self.d_model = d_model

        self.vocab_size = 4 ** kmer
        self.tok_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=None)
        starts = np.arange(0, seq_len - kmer + 1, stride)
        self.n_tok = len(starts)
        #                 k=4 -> [99,103)=R-A-C-H; k=6 -> [98,104)。
        self.center_pos = seq_len // 2
        off = (kmer - 1) // 2 if center_offset is None or center_offset < 0 else center_offset
        self.center_offset = off
        target_start = self.center_pos - off
        self.center_idx = int(np.argmin(np.abs(starts - target_start)))
        w = torch.tensor([4 ** (kmer - 1 - i) for i in range(kmer)], dtype=torch.long)
        self.register_buffer("tok_w", w, persistent=False)

        self.pos_mode = pos_mode
        if pos_mode == "none":
            self.pos_emb = None
        elif pos_mode == "sin":
            pe = torch.zeros(self.n_tok + 1, d_model)
            pos = torch.arange(self.n_tok + 1, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                            * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pos_table", pe, persistent=False)
            self.pos_emb = None
        elif pos_mode == "abs":
            self.pos_emb = nn.Embedding(self.n_tok + 1, d_model)
            ids = torch.arange(self.n_tok + 1, dtype=torch.long)
            self.register_buffer("pos_ids", ids, persistent=False)
        elif pos_mode == "center_sym":
            rel = torch.arange(self.n_tok, dtype=torch.long) - self.center_idx
            ids = torch.cat([torch.zeros(1, dtype=torch.long), rel.abs() + 1])
            self.max_dist = int(rel.abs().max().item())
            self.pos_emb = nn.Embedding(self.max_dist + 2, d_model)
            self.register_buffer("pos_ids", ids, persistent=False)
        else:
            raise ValueError("unknown pos_mode: %s" % pos_mode)

        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(d_model)

        nn.init.normal_(self.cls_tok, std=0.02)
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    def tokenize(self, idx):
        win = idx.unfold(1, self.kmer, self.stride)          # [B, n_tok, k]
        return (win * self.tok_w).sum(-1)

    def forward(self, idx, need_attn=False):
        h = self.tok_emb(self.tokenize(idx))                 # [B, n_tok, D]
        cls = self.cls_tok.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)                       # [B, n_tok+1, D]

        if self.pos_emb is not None:
            h = h + self.pos_emb(self.pos_ids)
        elif self.pos_mode == "sin":
            h = h + self.pos_table.unsqueeze(0)

        h = self.emb_drop(self.emb_norm(h))
        h = h.transpose(0, 1)
        if need_attn:
            h = self.encoder(h, mask=None)
        else:
            h = self.encoder(h)
        h = h.transpose(0, 1)                                # [B, L, D]
        return self.final_norm(h)


class RNAm6AClassifier(nn.Module):

    def __init__(self, seq_len=201, kmer=4, stride=1, d_model=96, nhead=4,
                 layers=4, dim_ff=192, dropout=0.15, pos_mode="abs", pool="center",
                 head_hidden=None, center_offset=-1):
        super().__init__()
        self.encoder = RNAFormerEncoder(seq_len=seq_len, kmer=kmer, stride=stride,
                                        d_model=d_model, nhead=nhead, layers=layers,
                                        dim_ff=dim_ff, dropout=dropout, pos_mode=pos_mode,
                                        center_offset=center_offset)
        self.pool = pool
        self.d_model = d_model

        if pool == "attn":
            self.pool_q = nn.Parameter(torch.zeros(d_model))
            nn.init.normal_(self.pool_q, std=0.02)

        head_hidden = head_hidden or max(32, d_model // 2)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, idx):
        h = self.encoder(idx)                                # [B, n_tok+1, D]
        if self.pool == "cls":
            feat = h[:, 0]
        elif self.pool == "center":
            feat = h[:, self.encoder.center_idx + 1]
        elif self.pool == "mean":
            feat = h[:, 1:].mean(dim=1)
        elif self.pool == "attn":
            score = (h * self.pool_q).sum(-1) / (self.d_model ** 0.5)
            weight = torch.softmax(score, dim=1)
            feat = (h * weight.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError("unknown pool: %s" % self.pool)
        return self.head(feat)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(mode="auto", log=None):
    def _log(m):
        if log:
            log(m)
        else:
            print(m, flush=True)

    if mode == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        _log("no CUDA available, using CPU")
        return torch.device("cpu")

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    sm = "sm_%d%d" % (cap[0], cap[1])
    try:
        arch = torch.cuda.get_arch_list()
    except Exception:
        arch = []
    ok = (not arch) or (sm in arch) or ("compute_%d%d" % cap in arch)
    if not ok:
        _log("device %s (%s) not in the arch list %s of torch(%s),"
             "falling back to CPU to avoid a hang."
             % (name, sm, torch.__version__, arch))
        _log("   for GPU training install torch cu128.")
        return torch.device("cpu")
    _log("using GPU: %s (%s)" % (name, sm))
    return torch.device("cuda")


def calc_all_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / den if den > 0 else 0.0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) > 0 else 0.0
    return {"AUC": float(roc_auc_score(y_true, y_prob)),
            "AP": float(average_precision_score(y_true, y_prob)),
            "MCC": float(mcc), "Acc": float(acc), "F1": float(f1),
            "Sen(Recall)": float(sen), "Spe": float(spe),
            "Precision": float(pre), "threshold": float(threshold)}


def best_threshold_by_youden(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def plot_curves(preds, save_roc, save_pr, title=""):
    plt.figure(figsize=(6, 5))
    for yt, yp, lab in preds:
        fpr, tpr, _ = roc_curve(yt, yp)
        plt.plot(fpr, tpr, lw=1.6, label="%s AUC=%.4f" % (lab, roc_auc_score(yt, yp)))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC %s" % title); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(save_roc, dpi=300); plt.close()

    plt.figure(figsize=(6, 5))
    for yt, yp, lab in preds:
        prec, rec, _ = precision_recall_curve(yt, yp)
        plt.plot(rec, prec, lw=1.6, label="%s AP=%.4f" % (lab, average_precision_score(yt, yp)))
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR %s" % title); plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(save_pr, dpi=300); plt.close()


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, verbose=True):
        self.patience, self.delta, self.verbose = patience, delta, verbose
        self.counter, self.best = 0, -1.0
        self.should_stop = False

    def step(self, val_auc):
        if val_auc > self.best + self.delta:
            self.best = val_auc
            self.counter = 0
            return True, False
        self.counter += 1
        if self.verbose:
            print("    no val-AUC improvement %d/%d" % (self.counter, self.patience))
        if self.counter >= self.patience:
            self.should_stop = True
        return False, self.should_stop


def build_scheduler(optimizer, total_steps, warmup_ratio, kind="cosine"):
    warmup = max(1, int(total_steps * warmup_ratio))
    if kind == "none":
        return None

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def run_once(args, device, log, tr_x, tr_y, va_x, va_y, te_x, te_y, fold_tag="", seed=None):
    seed = args.seed if seed is None else seed
    set_seed(seed)

    tr_ds = RNAm6ADataset(tr_x, tr_y, augment_shift=args.aug_mask)
    va_ds = RNAm6ADataset(va_x, va_y)
    te_ds = RNAm6ADataset(te_x, te_y)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers)

    model = RNAm6AClassifier(
        seq_len=args.seq_len, kmer=args.kmer, stride=args.stride,
        d_model=args.d_model, nhead=args.nhead, layers=args.layers,
        dim_ff=args.dim_ff, dropout=args.dropout, pos_mode=args.pos_mode,
        pool=args.pool, head_hidden=args.head_hidden,
        center_offset=args.center_offset).to(device)

    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)

    n_param = model.num_params()

    pw = None
    if args.pos_weight and args.pos_weight > 0:
        pw = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = max(1, len(tr_ld)) * args.epochs
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio, args.sched)

    stopper = EarlyStopping(patience=args.patience, verbose=args.verbose)
    best_state, best_val_auc, best_epoch = None, -1.0, -1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            if args.label_smooth > 0:
                by = by * (1 - args.label_smooth) + 0.5 * args.label_smooth
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            tot_loss += loss.item() * bx.size(0)
        avg_loss = tot_loss / max(len(tr_ds), 1)

        model.eval()
        val_true, val_prob = [], []
        with torch.no_grad():
            for bx, by in va_ld:
                p = torch.sigmoid(model(bx.to(device))).cpu().numpy().ravel()
                val_prob.extend(p.tolist())
                val_true.extend(by.numpy().ravel().tolist())
        val_auc = roc_auc_score(val_true, val_prob)
        history.append({"epoch": epoch, "loss": avg_loss, "val_auc": float(val_auc)})

        improved, stop = stopper.step(val_auc)
        if improved:
            best_val_auc, best_epoch = val_auc, epoch
            best_state = deepcopy(model.state_dict())
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(loader):
        yt, yp = [], []
        with torch.no_grad():
            for bx, by in loader:
                yt.extend(by.numpy().ravel().tolist())
                yp.extend(torch.sigmoid(model(bx.to(device))).cpu().numpy().ravel().tolist())
        return np.array(yt), np.array(yp)

    y_va, p_va = predict(va_ld)
    y_te, p_te = predict(te_ld)

    thr = 0.5
    if args.tune_threshold:
        thr = best_threshold_by_youden(y_va, p_va)
    m_val = calc_all_metrics(y_va, p_va, thr)
    m_test = calc_all_metrics(y_te, p_te, thr)

    return {"model": model, "state_dict": model.state_dict(),
            "val_metrics": m_val, "test_metrics": m_test,
            "val_pred": (y_va, p_va), "test_pred": (y_te, p_te),
            "best_epoch": best_epoch, "best_val_auc": best_val_auc,
            "n_param": n_param, "history": history, "threshold": thr}


def _fmt(m):
    return ("AUC=%.4f AP=%.4f MCC=%.4f Acc=%.4f F1=%.4f Sen=%.4f Spe=%.4f (thr=%.3f)"
            % (m["AUC"], m["AP"], m["MCC"], m["Acc"], m["F1"],
               m["Sen(Recall)"], m["Spe"], m["threshold"]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plain Transformer for m6A site prediction (no convolution)")
    p.add_argument("--cell", type=str, default="liver", help="cell line; reads <data_dir>/{cell}_*.csv")
    p.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    p.add_argument("--out_dir", type=str, default=str(BASE_DIR / "output_trans"))
    p.add_argument("--model_dir", type=str, default=str(BASE_DIR / "models"))
    p.add_argument("--tag", type=str, default="", help="suffix for output files")
    p.add_argument("--no_save", action="store_true", help="train only, save nothing (smoke test)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--sched", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--label_smooth", type=float, default=0.0)
    p.add_argument("--pos_weight", type=float, default=0.0, help=">0: enable BCE pos_weight (data are balanced, off by default)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val_ratio", type=float, default=0.1, help="validation fraction of the training split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=1, help=">1: stratified CV, report mean +/- sd")
    p.add_argument("--tune_threshold", action="store_true", help="choose the threshold by Youden index on validation")
    p.add_argument("--aug_mask", type=int, default=0, help="1: 5% random mutation outside centre +/- 10 nt")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=201)
    p.add_argument("--kmer", type=int, default=4, help="token length: 1/3/4/6")
    p.add_argument("--stride", type=int, default=1, help="k-mer stride (>1: non-overlapping)")
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim_ff", type=int, default=192)
    p.add_argument("--head_hidden", type=int, default=0, help="0: auto = max(32, d_model//2)")
    p.add_argument("--pool", type=str, default="center", choices=["cls", "center", "mean", "attn"])
    p.add_argument("--center_offset", type=int, default=-1,
                   help="index of the site inside the centre k-mer token; -1: auto (k-1)//2"
                        ">0: window-alignment ablation (0 = window starts at the site)")
    p.add_argument("--pos_mode", type=str, default="abs", choices=["abs", "center_sym", "sin", "none"])
    p.add_argument("--init_ckpt", type=str, default="", help="load pretrained encoder weights")
    p.add_argument("--max_train", type=int, default=0, help=">0: use only the first N training samples")
    p.add_argument("--max_test", type=int, default=0, help=">0: use only the first N test samples")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.head_hidden = args.head_hidden or None

    cell = args.cell
    out_dir = Path(args.out_dir) / ("%s%s" % (cell, args.tag))
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.txt" if not args.no_save else None

    log_fp = open(log_path, "w", encoding="utf-8") if log_path else None

    def log(msg):
        print(msg, flush=True)
        if log_fp:
            log_fp.write(msg + "\n")
            log_fp.flush()

    device = pick_device(args.device, log=log)


    tr_csv = Path(args.data_dir) / ("%s_train.csv" % cell)
    te_csv = Path(args.data_dir) / ("%s_test.csv" % cell)
    if not tr_csv.exists():
        raise FileNotFoundError("training set not found: %s" % tr_csv)
    df_tr = pd.read_csv(tr_csv)
    df_te = pd.read_csv(te_csv)

    tr_seq = df_tr["seq"].astype(str).tolist()
    tr_lab = df_tr["label"].astype(int).values
    te_seq = df_te["seq"].astype(str).tolist()
    te_lab = df_te["label"].astype(int).values
    if args.max_train > 0:
        tr_seq, tr_lab = tr_seq[:args.max_train], tr_lab[:args.max_train]
    if args.max_test > 0:
        te_seq, te_lab = te_seq[:args.max_test], te_lab[:args.max_test]

    len_counts = pd.Series([len(s) for s in tr_seq[:2000]]).value_counts().to_dict()

    t0 = time.time()
    tr_x = encode_sequences(tr_seq, args.seq_len)
    te_x = encode_sequences(te_seq, args.seq_len)

    if args.folds and args.folds > 1:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_results, fold_preds = [], []
        for k, (tr_idx, va_idx) in enumerate(skf.split(tr_x, tr_lab), start=1):
            tag = "[Fold %d/%d]" % (k, args.folds)
            res = run_once(args, device, log,
                           tr_x[tr_idx], tr_lab[tr_idx],
                           tr_x[va_idx], tr_lab[va_idx],
                           te_x, te_lab, fold_tag=tag + " ", seed=args.seed + k)
            fold_results.append(res)
            fold_preds.append((res["test_pred"][0], res["test_pred"][1], "Fold%d" % k))

        agg = {}
        for key in ("AUC", "AP", "MCC", "Acc", "F1", "Sen(Recall)", "Spe", "Precision"):
            vals = np.array([r["test_metrics"][key] for r in fold_results])
            agg[key] = (float(vals.mean()), float(vals.std()))
        res_main = fold_results[0]
        if not args.no_save:
            pd.DataFrame([{ "cell": cell, "fold": i + 1, **{k: v for k, v in r["test_metrics"].items()}}
                          for i, r in enumerate(fold_results)]
                         ).to_csv(out_dir / "metrics_per_fold.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([{ "cell": cell, "folds": args.folds,
                            **{k: "%.4f±%.4f" % v for k, v in agg.items()}}
                          ]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
            plot_curves(fold_preds, out_dir / "roc.png", out_dir / "pr.png",
                        title="%s (%d-fold, pure Transformer)" % (cell, args.folds))
        final_metrics = {k: v[0] for k, v in agg.items()}
        final_std = {k: v[1] for k, v in agg.items()}
    else:
        tr_idx, va_idx = train_test_split(
            np.arange(len(tr_lab)), test_size=args.val_ratio,
            random_state=args.seed, stratify=tr_lab)
        res_main = run_once(args, device, log,
                            tr_x[tr_idx], tr_lab[tr_idx],
                            tr_x[va_idx], tr_lab[va_idx],
                            te_x, te_lab, fold_tag="", seed=args.seed)
        final_metrics = res_main["test_metrics"]
        final_std = {k: 0.0 for k in final_metrics}
        if not args.no_save:
            pd.DataFrame([{"cell": cell, **final_metrics}]).to_csv(
                out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
            plot_curves([(res_main["test_pred"][0], res_main["test_pred"][1], cell)],
                        out_dir / "roc.png", out_dir / "pr.png",
                        title="%s (pure Transformer)" % cell)

    if not args.no_save:
        model_path = Path(args.model_dir) / ("%s_trans%s.pkl" % (cell, args.tag))
        torch.save({"state_dict": res_main["state_dict"],
                    "args": vars(args),
                    "test_metrics": final_metrics}, model_path)
        with open(out_dir / "train_log_final.json", "w", encoding="utf-8") as f:
            json.dump({"args": vars(args),
                       "test_metrics": final_metrics,
                       "test_metrics_std": final_std,
                       "best_epoch": res_main["best_epoch"],
                       "best_val_auc": res_main["best_val_auc"],
                       "n_param": res_main["n_param"],
                       "history": res_main["history"]}, f, ensure_ascii=False, indent=2)

    if log_fp:
        log_fp.close()
    return final_metrics


if __name__ == "__main__":
    main()
    print("[done] see metrics.csv / metrics_per_fold.csv under --out_dir "
          "(default: output_trans/)")
