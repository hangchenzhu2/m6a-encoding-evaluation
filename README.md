# Input Encoding Outweighs Network Architecture in RNA m6A Site Prediction

Code and data for *Input Encoding Outweighs Network Architecture in RNA m6A Site
Prediction: A Cross-Dataset and Cross-Architecture Evaluation*.

## Contents

```
data/original/              11 benchmark datasets (train + test)
data/rebuilt/               12 datasets rebuilt in this work (train + test)
models/                     ENAC-16 weights, one per dataset (fold 1, seed 42)
expected_results.csv        reference AUC for every dataset x encoding
train_encode_ablation.py    main training: 6 encodings x 6 model arms x 5 folds
predict.py                  score sequences with the released ENAC-16 weights
train_trans.py              Transformer backbone
classic_ml_compare.py       classical ML baselines
ism_attribution.py          in-silico mutagenesis attribution
build_dataset.py            rebuild the 12 datasets from RMBase + GRCh38
attention_blocks.py         SE / ECA / CBAM blocks
```

Each CSV has the columns `chr, loc, strand, seq, label`. Sequences are 201 nt,
the modified adenine is at 0-based index 100, positives and negatives are
balanced 1:1.

## Install

```
pip install -r requirements.txt
```

## Run

```
python train_encode_ablation.py --cell liver        # 6 encodings, CNN, 5 folds
python train_trans.py --cell liver --kmer 3         # Transformer backbone
python classic_ml_compare.py --datasets liver kidney A549
python ism_attribution.py --cell liver --encoding enac_onehot16
python predict.py --model models/original_liver_enac_onehot16.pkl 
                  --data data/original/liver_test.csv        # AUC / ACC / MCC
```

Quick check (about 10 s):
`python train_encode_ablation.py --cell liver --folds 1 --epochs 1`

## Output

Every script creates its output directory. `results.csv` holds one row per run
with `encoding, model, seed, AUC, AP, MCC, Acc, Sen, Spe, fold`; figures go to
`pic/`. Compare your numbers with `expected_results.csv`.

## Notes

- Fold *i* uses seed `42 + i`. Cross-validation runs inside the training split
  only; every fold is scored on the same held-out test set.
- The weights in `models/` are from the first fold (seed 42). The paper reports
  five-fold means, so a single model lands within roughly the fold-to-fold
  standard deviation given in `expected_results.csv`.
- `data/rebuilt/` holds 4 datasets that reuse benchmark cell-line names
  (HeLa, HEK293T, A549, HepG2) but are rebuilt from an independent source, and
  8 from cell types absent from the benchmark.
- `build_dataset.py` needs the RMBase v3.0 hg38 BED file and the Ensembl GRCh38
  assembly plus release-110 GTF; the resulting CSVs are already in
  `data/rebuilt/`.

## Data source
- `data/original/` — the 11 benchmark datasets released with CLSM6A
  (Zhang et al., *Bioinformatics* 2023;39(12):btad709).
- `data/rebuilt/` — the 12 datasets built in this work from RMBase v3.0
  (Xuan et al., *Nucleic Acids Res* 2024;52(D1):D273-D284) and Ensembl
  GRCh38 release 110 (Martin et al., *Nucleic Acids Res* 2023;51(D1):D933-D941).

The 11 datasets in `data/original/` are the benchmark released with CLSM6A;
please cite Zhang et al., *Bioinformatics* 2023;39(12):btad709.
