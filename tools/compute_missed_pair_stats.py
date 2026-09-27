import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from collections import Counter

with open("artifacts/reports/blocking_decomposition.json") as f:
    dec = json.load(f)

# Let's run a quick calculation of the feature distributions over missed links
import yaml, csv
from pathlib import Path
from main import load_source_records, load_ground_truth, NormalizedRecord
from rapidfuzz import fuzz

with open("config.yaml") as f:
    config = yaml.safe_load(f)

train_dir = Path(config["paths"]["train_dir"])
s1_all = load_source_records(train_dir / "train_source1.tsv", max_rows=60000)
s1_keys = list(s1_all.keys())
rng = np.random.default_rng(42)
shuffled_keys = list(s1_keys)
rng.shuffle(shuffled_keys)
val_s1_ids = set(shuffled_keys[:12000])

gt_all = load_ground_truth(train_dir / "train_ground_truth.tsv", filter_s1_ids=set(s1_keys))
val_gt = {k: gt_all.get(k, set()) for k in val_s1_ids}

needed_pos = set()
for s in val_s1_ids:
    needed_pos |= val_gt.get(s, set())

s2 = load_source_records(train_dir / "train_source2.tsv", max_rows=300000)
s3 = load_source_records(train_dir / "train_source3.tsv", max_rows=300000)
targets = {**s2, **s3}

missing_pos = needed_pos - set(targets.keys())
if missing_pos:
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        if not missing_pos:
            break
        with (train_dir / fname).open("r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                if len(row) >= 4 and row[0] in missing_pos:
                    rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                    targets[rec.entity_id] = rec
                    missing_pos.remove(rec.entity_id)

# Build baseline candidates to get exact 10,339 missed links
from tools.test_blocker_experiments import main as _
# Let's read missed pairs from baseline
# We can reproduce baseline easily
targets_by_c = {}
for r in targets.values():
    targets_by_c.setdefault(r.country, []).append(r)

b_idx = {}
for c, tlist in targets_by_c.items():
    b_idx[c] = {
        "en": {}, "sn": {}, "ea": {}, "na": {}, "rt": {}, "cp": {}
    }
    # name token freqs
    tf = {}
    for r in tlist:
        for t in set(r.name_tokens):
            if len(t) >= 3 and t not in {"ltd", "pvt", "inc", "corp", "co", "llc"}:
                tf[t] = tf.get(t, 0) + 1
    for r in tlist:
        eid = r.entity_id
        if r.name_norm and len(b_idx[c]["en"].setdefault(r.name_norm, [])) < 60:
            b_idx[c]["en"][r.name_norm].append(eid)
        if r.name_sorted != r.name_norm and len(b_idx[c]["sn"].setdefault(r.name_sorted, [])) < 60:
            b_idx[c]["sn"][r.name_sorted].append(eid)
        if not r.addr_missing and len(r.addr_norm) >= 8 and len(b_idx[c]["ea"].setdefault(r.addr_norm, [])) < 60:
            b_idx[c]["ea"][r.addr_norm].append(eid)
        if r.name_tokens and r.addr_numbers:
            k = f"{r.name_tokens[0]}_{r.addr_numbers[0]}"
            if len(b_idx[c]["na"].setdefault(k, [])) < 60:
                b_idx[c]["na"][k].append(eid)
        best_t, best_f = None, 999999
        for t in r.name_tokens:
            f = tf.get(t, 0)
            if 1 <= f <= 50 and f < best_f:
                best_f, best_t = f, t
        if best_t and len(b_idx[c]["rt"].setdefault(best_t, [])) < 60:
            b_idx[c]["rt"][best_t].append(eid)
        if r.name_norm and r.addr_numbers and len(r.name_norm) >= 4:
            k = f"{r.name_norm[:4]}_{r.addr_numbers[0]}"
            if len(b_idx[c]["cp"].setdefault(k, [])) < 60:
                b_idx[c]["cp"][k].append(eid)

missed = []
for s1_id in val_s1_ids:
    s1 = s1_all[s1_id]
    c = s1.country
    bi = b_idx.get(c, {})
    cands = set()
    if s1.name_norm:
        cands.update(bi["en"].get(s1.name_norm, []))
    if s1.name_sorted != s1.name_norm:
        cands.update(bi["sn"].get(s1.name_sorted, []))
    if not s1.addr_missing and len(s1.addr_norm) >= 8:
        cands.update(bi["ea"].get(s1.addr_norm, []))
    if s1.name_tokens and s1.addr_numbers:
        k = f"{s1.name_tokens[0]}_{s1.addr_numbers[0]}"
        cands.update(bi["na"].get(k, []))
    for t in s1.name_tokens:
        cands.update(bi["rt"].get(t, []))
    if s1.name_norm and s1.addr_numbers and len(s1.name_norm) >= 4:
        k = f"{s1.name_norm[:4]}_{s1.addr_numbers[0]}"
        cands.update(bi["cp"].get(k, []))

    for tid in val_gt[s1_id]:
        if tid not in cands:
            missed.append((s1, targets[tid]))

print(f"Total missed true pairs: {len(missed)}")

name_sims = [fuzz.ratio(s.name_norm, t.name_norm) for s, t in missed]
addr_sims = [fuzz.ratio(s.addr_norm, t.addr_norm) for s, t in missed if not s.addr_missing and not t.addr_missing]
s1_name_lens = [len(s.name_norm) for s, t in missed]
tgt_name_lens = [len(t.name_norm) for s, t in missed]
s1_addr_lens = [len(s.addr_norm) for s, t in missed]
tgt_addr_lens = [len(t.addr_norm) for s, t in missed]
tok_jaccards = [len(set(s.name_tokens) & set(t.name_tokens)) / max(1, len(set(s.name_tokens) | set(t.name_tokens))) for s, t in missed]
tok_overlaps = [len(set(s.name_tokens) & set(t.name_tokens)) / max(1, min(len(s.name_tokens), len(t.name_tokens))) for s, t in missed if s.name_tokens and t.name_tokens]

tgt_missing_addr = sum(1 for s, t in missed if t.addr_missing)
s1_missing_addr = sum(1 for s, t in missed if s.addr_missing)
num_overlaps = [len(set(s.addr_numbers) & set(t.addr_numbers)) for s, t in missed]

countries = Counter(s.country for s, t in missed)
sources = Counter(t.source for s, t in missed)

print("\n--- Summary Stats of Missed Pairs ---")
print(f"Name Sim (Levenshtein): Mean={np.mean(name_sims):.1f}, P10={np.percentile(name_sims, 10):.1f}, P50={np.median(name_sims):.1f}, P90={np.percentile(name_sims, 90):.1f}")
print(f"Addr Sim (Non-missing): Mean={np.mean(addr_sims):.1f}, P10={np.percentile(addr_sims, 10):.1f}, P50={np.median(addr_sims):.1f}, P90={np.percentile(addr_sims, 90):.1f}")
print(f"Token Jaccard: Mean={np.mean(tok_jaccards):.3f}, P50={np.median(tok_jaccards):.3f}, P90={np.percentile(tok_jaccards, 90):.3f}")
print(f"Token Overlap: Mean={np.mean(tok_overlaps):.3f}, P50={np.median(tok_overlaps):.3f}, P90={np.percentile(tok_overlaps, 90):.3f}")
print(f"Numeric Overlap: Mean={np.mean(num_overlaps):.2f}, Pairs with >=1 shared num={sum(1 for n in num_overlaps if n > 0)} ({sum(1 for n in num_overlaps if n > 0)/len(missed)*100:.1f}%)")
print(f"S1 Name Len: Mean={np.mean(s1_name_lens):.1f} | Tgt Name Len: Mean={np.mean(tgt_name_lens):.1f}")
print(f"S1 Addr Len: Mean={np.mean(s1_addr_lens):.1f} | Tgt Addr Len: Mean={np.mean(tgt_addr_lens):.1f}")
print(f"Tgt Addr Missing: {tgt_missing_addr} ({tgt_missing_addr/len(missed)*100:.2f}%) | S1 Addr Missing: {s1_missing_addr} ({s1_missing_addr/len(missed)*100:.2f}%)")
print(f"Countries: {dict(countries)}")
print(f"Sources: {dict(sources)}")
