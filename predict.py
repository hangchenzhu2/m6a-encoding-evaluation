# -*- coding: utf-8 -*-
"""
predict.py —— Load a trained ENAC-16 model and predict m6A sites.

Examples
--------
    # score a test CSV and report metrics
    python predict.py --model models/original_liver_enac_onehot16.pkl \
                      --data data/original/liver_test.csv

    # score a single 201-nt sequence
    python predict.py --model models/original_liver_enac_onehot16.pkl \
                      --seq AUGC...   # 201 nt

    # write per-sequence probabilities
    python predict.py --model models/original_liver_enac_onehot16.pkl \
                      --data data/original/liver_test.csv --out preds.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from attention_blocks import SEBlock1d, ECABlock1d, CBAMBlock1d
from train_encode_ablation import AblationCNN, encode_sequences, calc_metrics

ENCODING = "enac_onehot16"
IN_CHANNELS = 16


def build_model(state, device):
    """Rebuild the network that produced `state` and load the weights."""
    attn = "none"
    if any(k.startswith("a1.") for k in state):
        # attention arms store parameters under a1./a2./a3.
        n = sum(1 for k in state if k.startswith("a3."))
        attn = "se" if n else "cbam"
    model = AblationCNN(in_channels=IN_CHANNELS, attn=attn)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit("weight file does not match the model: %s" % unexpected[:4])
    return model.to(device).eval()


@torch.no_grad()
def predict_probs(model, seqs, device, batch=256):
    x = encode_sequences(list(seqs), ENCODING)
    if isinstance(x, tuple):
        x = x[0]
    x = torch.from_numpy(np.asarray(x)).float()
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch].to(device)).cpu().numpy().ravel())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description="Predict m6A sites with a trained ENAC-16 model")
    ap.add_argument("--model", required=True, help="path to a .pkl produced by train_encode_ablation.py")
    ap.add_argument("--data", default="", help="CSV with a seq column (and optionally label)")
    ap.add_argument("--seq", default="", help="a single 201-nt sequence")
    ap.add_argument("--out", default="", help="write per-sequence probabilities to this CSV")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit("model not found: %s" % args.model)
    if not args.data and not args.seq:
        raise SystemExit("give either --data or --seq")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    state = torch.load(args.model, map_location="cpu")
    model = build_model(state, device)
    print("[device] %s   [model] %s" % (device, Path(args.model).name))

    # ---------- single sequence ----------
    if args.seq:
        s = args.seq.strip().upper().replace("T", "U")
        if len(s) != 201:
            raise SystemExit("sequence must be 201 nt, got %d" % len(s))
        p = float(predict_probs(model, [s], device)[0])
        print("[result] P(m6A) = %.4f   ->  %s (threshold 0.5)"
              % (p, "modified" if p >= 0.5 else "not modified"))
        return

    # ---------- CSV ----------
    df = pd.read_csv(args.data)
    if "seq" not in df.columns:
        raise SystemExit("CSV must have a 'seq' column")
    seqs = df["seq"].astype(str).str.strip().str.upper().tolist()
    probs = predict_probs(model, seqs, device)

    out = pd.DataFrame({"seq": seqs, "prob": probs})
    if "label" in df.columns:
        y = df["label"].astype(int).values
        m = calc_metrics(y, probs)
        out["label"] = y
        print("[metrics] " + "  ".join("%s=%.4f" % (k, v) for k, v in m.items()))
    else:
        print("[result] scored %d sequences, mean P(m6A) = %.4f"
              % (len(probs), float(np.mean(probs))))

    if args.out:
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
        print("[saved] %s" % args.out)


if __name__ == "__main__":
    main()
