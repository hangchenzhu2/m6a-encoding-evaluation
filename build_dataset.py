# -*- coding: utf-8 -*-
import argparse
import bisect
import gzip
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
GENOME = BASE / "_genome"
BED = BASE / "test" / "hg38.m6A" / "human.hg38.m6A.result.col29.bed"

SPACING = 200
FLANK = 100
SEED = 42

D = set("AGT")
R = set("AG")
H = set("ACT")

RE_PLUS = re.compile("[AGT][AG]AC[ACT]")
#   A->T, C->G, R(A/G)->(T/C), D(A/G/T)->(T/C/A), H(A/C/T)->(T/G/A)
RE_MINUS = re.compile("[TGA]GT[TC][TCA]")

_COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")


def revcomp(s):
    return s.translate(_COMP)[::-1]


def load_chrom(name):
    p = GENOME / ("Homo_sapiens.GRCh38.dna.chromosome.%s.fa.gz" % name)
    if not p.is_file():
        return None
    parts = []
    with gzip.open(p, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                continue
            parts.append(line.strip())
    return "".join(parts).upper()


def read_positives(cell):
    out = defaultdict(list)
    skipped_scaffold = 0
    total = 0
    with open(BED, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 19:
                continue
            if f[11].strip() != cell:
                continue
            s = f[18].upper()
            if len(s) != 41:
                continue
            if not (s[18] in D and s[19] in R and s[20] == "A"
                    and s[21] == "C" and s[22] in H):
                continue
            total += 1
            chrom = f[0]
            if not chrom.startswith("chr") or chrom[3:] not in (
                    [str(i) for i in range(1, 23)] + ["X", "Y"]):
                skipped_scaffold += 1
                continue
            try:
                out[chrom[3:]].append((int(f[1]), f[5], s[18:23]))
            except Exception:
                pass
    return out, total, skipped_scaffold


def thin(positions):
    positions = sorted(positions)
    kept = []
    last = -10 ** 9
    for p in positions:
        if p - last >= SPACING:
            kept.append(p)
            last = p
    return kept


EXONS = {}


def load_exons():
    import gzip as _gz
    gtf = GENOME / "Homo_sapiens.GRCh38.110.gtf.gz"
    if not gtf.is_file():
        return False
    raw = defaultdict(list)
    with _gz.open(gtf, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t")
            if len(f) < 5 or f[2] != "exon":
                continue
            raw[f[0]].append((int(f[3]) - 1, int(f[4])))
    for c, iv in raw.items():
        if c not in [str(i) for i in range(1, 23)] + ["X", "Y"]:
            continue
        iv.sort()
        m = []
        for s, e in iv:
            if m and s <= m[-1][1]:
                m[-1] = (m[-1][0], max(m[-1][1], e))
            else:
                m.append((s, e))
        EXONS[c] = ([x[0] for x in m], [x[1] for x in m])
    return True


def exon_ok(chrom, pos0):
    if not EXONS:
        return True
    if chrom not in EXONS:
        return False
    starts, ends = EXONS[chrom]
    i = bisect.bisect_right(starts, pos0) - 1
    return i >= 0 and pos0 < ends[i]


def main():
    ap = argparse.ArgumentParser(description="Build an m6A dataset")
    ap.add_argument("--cell", required=True, help="cell line; must match the cell column of the BED file")
    ap.add_argument("--out_dir", default=str(BASE / "new_data"))
    ap.add_argument("--spacing", type=int, default=SPACING)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chroms", default="", help="restrict to these chromosomes (comma separated; empty = all)")
    ap.add_argument("--max_pos", type=int, default=0,
                    help="max number of positives (0 = unlimited)")
    args = ap.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)

    def log(m=""):
        print(m, flush=True)


    pos_by_chr, n_total, n_skipped = read_positives(args.cell)

    if args.chroms:
        keep = [x.strip() for x in args.chroms.split(",") if x.strip()]
        pos_by_chr = {c: v for c, v in pos_by_chr.items() if c in keep}

    pos_thin = {}
    n_after = 0
    for c, recs in pos_by_chr.items():
        kept = thin([r[0] for r in recs])
        keep_set = set(kept)
        pos_thin[c] = [r for r in recs if r[0] in keep_set]
        n_after += len(pos_thin[c])

    load_exons()

    positives = []      # (chrom, pos0, strand, seq)
    negatives = []
    for idx, cname in enumerate(sorted(pos_thin, key=lambda x: (len(x), x)), 1):
        recs = pos_thin[cname]
        if not recs:
            continue
        genome = load_chrom(cname)
        if genome is None:
            continue
        L = len(genome)
        pos_sorted = sorted(s[0] for s in recs)

        got_pos = 0
        for p0, strand, _ in recs:
            if p0 - FLANK < 0 or p0 + FLANK + 1 > L:
                continue
            w = genome[p0 - FLANK:p0 + FLANK + 1]
            if strand == "-":
                w = revcomp(w)
            if len(w) != 2 * FLANK + 1 or w[FLANK] != "A":
                continue
            positives.append((cname, p0, strand, w))
            got_pos += 1

        import bisect
        cand = []           # (pos0, strand)
        for m in RE_PLUS.finditer(genome):
            p0 = m.start() + 2
            i = bisect.bisect_left(pos_sorted, p0)
            near = False
            for j in (i - 1, i):
                if 0 <= j < len(pos_sorted) and abs(pos_sorted[j] - p0) < args.spacing:
                    near = True
                    break
            if not near and exon_ok(cname, p0):
                cand.append((p0, "+"))
        for m in RE_MINUS.finditer(genome):
            p0 = m.start() + 2
            i = bisect.bisect_left(pos_sorted, p0)
            near = False
            for j in (i - 1, i):
                if 0 <= j < len(pos_sorted) and abs(pos_sorted[j] - p0) < args.spacing:
                    near = True
                    break
            if not near and exon_ok(cname, p0):
                cand.append((p0, "-"))

        by_strand = defaultdict(list)
        for p0, st in cand:
            by_strand[st].append(p0)
        cand2 = []
        for st, ps in by_strand.items():
            cand2.extend((p, st) for p in thin(ps))

        rng.shuffle(cand2)
        got_neg = 0
        for p0, st in cand2:
            if got_neg >= got_pos:
                break
            if p0 - FLANK < 0 or p0 + FLANK + 1 > L:
                continue
            w = genome[p0 - FLANK:p0 + FLANK + 1]
            if st == "-":
                w = revcomp(w)
            if len(w) != 2 * FLANK + 1 or w[FLANK] != "A":
                continue
            negatives.append((cname, p0, st, w))
            got_neg += 1

        del genome


    for name, arr in [("pos", positives), ("neg", negatives)]:
        bad = sum(1 for r in arr if r[3][FLANK] != "A")
    seen = set()
    uniq_pos, uniq_neg = [], []
    for r in positives:
        if r[3] not in seen:
            seen.add(r[3]); uniq_pos.append(r)
    dup_pos = len(positives) - len(uniq_pos)
    seen = set()
    for r in negatives:
        if r[3] not in seen:
            seen.add(r[3]); uniq_neg.append(r)
    dup_neg = len(negatives) - len(uniq_neg)

    n = min(len(uniq_pos), len(uniq_neg))
    rng.shuffle(uniq_pos)
    rng.shuffle(uniq_neg)
    pos_sel, neg_sel = uniq_pos[:n], uniq_neg[:n]
    if args.max_pos and len(pos_sel) > args.max_pos:
        pos_sel = pos_sel[:args.max_pos]
        neg_sel = neg_sel[:args.max_pos]
        n = args.max_pos
    rows = [dict(chr="chr" + c, loc=p + 1, strand=s, seq=w, label=1) for c, p, s, w in pos_sel]
    rows += [dict(chr="chr" + c, loc=p + 1, strand=s, seq=w, label=0) for c, p, s, w in neg_sel]
    rng.shuffle(rows)
    n_train = int(len(rows) * 0.9)
    train, test = rows[:n_train], rows[n_train:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(train).to_csv(out / ("%s_train.csv" % args.cell), index=False, encoding="utf-8-sig")
    pd.DataFrame(test).to_csv(out / ("%s_test.csv" % args.cell), index=False, encoding="utf-8-sig")


    def five(r):
        w = r["seq"]
        return w[FLANK - 2:FLANK + 3]
    cp = Counter(five(r) for r in rows if r["label"] == 1)
    cn = Counter(five(r) for r in rows if r["label"] == 0)


if __name__ == "__main__":
    main()
