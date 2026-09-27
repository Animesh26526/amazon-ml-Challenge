#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Controlled Blocker Experiments

Tests targeted candidate blocker additions to resolve the top failure modes:
1. all_rare_tokens: Index all tokens with df <= 50 (instead of just the single rarest)
2. core_name_exact: Suffix-stripped name exact match
3. core_name_sorted: Suffix-stripped name sorted match
4. char_3_last_2: First 3 chars + last 2 chars of name
5. prefix_3_any_num: First 3 chars of name + any shared address number
6. rare_token_expanded: Tokens with df <= 100 or 200
7. addr_token_num: Rare address token + address number
8. token_intersection_bounded: Shared token with frequency <= 150

Measures:
- true links recovered
- candidate recall
- avg, median, P90, P95, P99, max candidates per S1
- candidate volume
- runtime
"""

import csv
import gc
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import yaml
from rapidfuzz import fuzz

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    COMMON_STOP_TOKENS,
    NormalizedRecord,
    load_ground_truth,
    load_source_records,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("blocker_experiments")

LEGAL_SUFFIXES_EXT = {
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "gmbh", "bv", "sa",
    "sas", "sarl", "spa", "pllc", "limited", "corporation", "incorporated",
    "private", "company", "cie", "nv", "ag", "sl", "srl",
}

def strip_legal_suffixes(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    """Strip trailing legal suffix tokens."""
    res = list(tokens)
    while res and res[-1] in LEGAL_SUFFIXES_EXT:
        res.pop()
    return tuple(res)


def main():
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    train_dir = Path(paths_cfg["train_dir"])
    artifact_dir = Path(paths_cfg["artifact_dir"])
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = config.get("experiment", {})
    train_sample_limit = exp_cfg.get("train_s1_sample", 60000)
    val_sample_limit = exp_cfg.get("val_s1_sample", 15000)
    val_ratio = exp_cfg.get("val_split_ratio", 0.20)
    seed = exp_cfg.get("seed", 42)

    logger.info("Loading S1 and Ground Truth...")
    s1_all = load_source_records(train_dir / "train_source1.tsv", max_rows=train_sample_limit)
    s1_keys = list(s1_all.keys())
    rng = np.random.default_rng(seed)
    shuffled_keys = list(s1_keys)
    rng.shuffle(shuffled_keys)

    n_val = min(val_sample_limit, int(len(shuffled_keys) * val_ratio))
    val_s1_ids = set(shuffled_keys[:n_val])

    gt_all = load_ground_truth(train_dir / "train_ground_truth.tsv", filter_s1_ids=set(s1_keys))
    val_gt = {k: gt_all.get(k, set()) for k in val_s1_ids}
    total_true_links = sum(len(v) for v in val_gt.values())
    logger.info(f"Loaded {len(val_s1_ids):,} Val S1 entities. Total true links: {total_true_links:,}")

    needed_positive_ids = set()
    for s1_id in val_s1_ids:
        needed_positive_ids |= val_gt.get(s1_id, set())

    # Load targets
    logger.info("Loading target records...")
    s2_records = load_source_records(train_dir / "train_source2.tsv", max_rows=300000)
    s3_records = load_source_records(train_dir / "train_source3.tsv", max_rows=300000)
    all_targets: Dict[str, NormalizedRecord] = {**s2_records, **s3_records}

    missing_pos_ids = needed_positive_ids - set(all_targets.keys())
    if missing_pos_ids:
        for fname in ["train_source2.tsv", "train_source3.tsv"]:
            if not missing_pos_ids:
                break
            with (train_dir / fname).open("r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r, None)
                for row in r:
                    if len(row) >= 4 and row[0] in missing_pos_ids:
                        rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                        all_targets[rec.entity_id] = rec
                        missing_pos_ids.remove(rec.entity_id)

    logger.info(f"Targets ready: {len(all_targets):,} records.")

    # Partition targets by country
    targets_by_country: Dict[str, List[NormalizedRecord]] = defaultdict(list)
    for r in all_targets.values():
        targets_by_country[r.country].append(r)

    # Document frequencies for target name tokens & address tokens
    name_token_freqs: Dict[str, Dict[str, int]] = {}
    addr_token_freqs: Dict[str, Dict[str, int]] = {}

    for country, tgt_list in targets_by_country.items():
        ntf = defaultdict(int)
        atf = defaultdict(int)
        for r in tgt_list:
            for t in set(r.name_tokens):
                if t not in COMMON_STOP_TOKENS and len(t) >= 3:
                    ntf[t] += 1
            for t in set(r.addr_tokens):
                if len(t) >= 4:
                    atf[t] += 1
        name_token_freqs[country] = ntf
        addr_token_freqs[country] = atf

    # Build Baseline MultiPassCandidateGenerator indexes
    logger.info("Building Baseline Candidate Generator...")
    block_cfg = config.get("blocking", {})
    max_candidates_per_key = 60
    max_candidates_per_s1 = 50

    base_indexes: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for country, tgt_list in targets_by_country.items():
        b_idx = {
            "exact_name": defaultdict(list),
            "sorted_name": defaultdict(list),
            "exact_addr": defaultdict(list),
            "numeric_anchor": defaultdict(list),
            "rare_token": defaultdict(list),
            "char_prefix": defaultdict(list),
        }
        tf = name_token_freqs[country]
        for r in tgt_list:
            eid = r.entity_id
            if r.name_norm and len(b_idx["exact_name"][r.name_norm]) < max_candidates_per_key:
                b_idx["exact_name"][r.name_norm].append(eid)
            if r.name_sorted != r.name_norm and len(b_idx["sorted_name"][r.name_sorted]) < max_candidates_per_key:
                b_idx["sorted_name"][r.name_sorted].append(eid)
            if not r.addr_missing and len(r.addr_norm) >= 8 and len(b_idx["exact_addr"][r.addr_norm]) < max_candidates_per_key:
                b_idx["exact_addr"][r.addr_norm].append(eid)
            if r.name_tokens and r.addr_numbers:
                num_key = f"{r.name_tokens[0]}_{r.addr_numbers[0]}"
                if len(b_idx["numeric_anchor"][num_key]) < max_candidates_per_key:
                    b_idx["numeric_anchor"][num_key].append(eid)
            # Baseline rarest token
            best_t = None
            best_f = 999999
            for t in r.name_tokens:
                f = tf.get(t, 0)
                if 1 <= f <= 50:
                    if f < best_f:
                        best_f = f
                        best_t = t
            if best_t and len(b_idx["rare_token"][best_t]) < max_candidates_per_key:
                b_idx["rare_token"][best_t].append(eid)
            if r.name_norm and r.addr_numbers and len(r.name_norm) >= 4:
                prefix_key = f"{r.name_norm[:4]}_{r.addr_numbers[0]}"
                if len(b_idx["char_prefix"][prefix_key]) < max_candidates_per_key:
                    b_idx["char_prefix"][prefix_key].append(eid)

        base_indexes[country] = b_idx

    # Baseline retrieval
    logger.info("Computing Baseline Candidates...")
    t0 = time.time()
    baseline_cands_by_s1: Dict[str, Set[str]] = {}
    for s1_id in val_s1_ids:
        s1 = s1_all[s1_id]
        country = s1.country
        b_idx = base_indexes.get(country, {})
        cands = defaultdict(int)

        if s1.name_norm:
            for tid in b_idx.get("exact_name", {}).get(s1.name_norm, []):
                cands[tid] += 1
        if s1.name_sorted != s1.name_norm:
            for tid in b_idx.get("sorted_name", {}).get(s1.name_sorted, []):
                cands[tid] += 1
        if not s1.addr_missing and len(s1.addr_norm) >= 8:
            for tid in b_idx.get("exact_addr", {}).get(s1.addr_norm, []):
                cands[tid] += 1
        if s1.name_tokens and s1.addr_numbers:
            num_key = f"{s1.name_tokens[0]}_{s1.addr_numbers[0]}"
            for tid in b_idx.get("numeric_anchor", {}).get(num_key, []):
                cands[tid] += 1
        for t in s1.name_tokens:
            if t not in COMMON_STOP_TOKENS and t in b_idx.get("rare_token", {}):
                for tid in b_idx["rare_token"][t]:
                    cands[tid] += 1
        if s1.name_norm and s1.addr_numbers and len(s1.name_norm) >= 4:
            prefix_key = f"{s1.name_norm[:4]}_{s1.addr_numbers[0]}"
            for tid in b_idx.get("char_prefix", {}).get(prefix_key, []):
                cands[tid] += 1

        if len(cands) > max_candidates_per_s1:
            sorted_items = sorted(cands.items(), key=lambda x: x[1], reverse=True)
            baseline_cands_by_s1[s1_id] = set(k for k, _ in sorted_items[:max_candidates_per_s1])
        else:
            baseline_cands_by_s1[s1_id] = set(cands.keys())

    baseline_time = time.time() - t0
    base_covered = sum(len(val_gt[s] & baseline_cands_by_s1[s]) for s in val_s1_ids)
    logger.info(f"Baseline: Covered = {base_covered:,} / {total_true_links:,} ({base_covered/total_true_links*100:.2f}%), Time={baseline_time:.2f}s")

    # ============================================================
    # BUILD INDIVIDUAL CANDIDATE EXPERIMENTAL ROUTES
    # ============================================================
    logger.info("Building experimental candidate blocker indices...")

    # We will build indices for each proposal:
    # Proposal 1: all_rare_tokens_df50 (Index ALL name tokens with df <= 50, not just the single rarest)
    # Proposal 2: core_name_exact (Suffix-stripped exact name)
    # Proposal 3: core_name_sorted (Suffix-stripped sorted name)
    # Proposal 4: prefix_3_any_num (First 3 chars of name + ANY address number)
    # Proposal 5: char_3_last_2 (First 3 chars + last 2 chars of core name)
    # Proposal 6: rare_tokens_df100 (Index ALL name tokens with df <= 100)
    # Proposal 7: addr_token_num (Rare address token df <= 50 + address number)
    # Proposal 8: core_token_prefix_num (First core token + ANY address number)

    exp_indices: Dict[str, Dict[str, Dict[str, List[str]]]] = {
        "all_rare_df50": {c: defaultdict(list) for c in targets_by_country},
        "core_name_exact": {c: defaultdict(list) for c in targets_by_country},
        "core_name_sorted": {c: defaultdict(list) for c in targets_by_country},
        "prefix_3_any_num": {c: defaultdict(list) for c in targets_by_country},
        "char_3_last_2": {c: defaultdict(list) for c in targets_by_country},
        "rare_tokens_df100": {c: defaultdict(list) for c in targets_by_country},
        "addr_token_num": {c: defaultdict(list) for c in targets_by_country},
        "core_prefix_num": {c: defaultdict(list) for c in targets_by_country},
    }

    # Policy on max candidates per key for experimental routes
    EXP_KEY_CAP = 60

    for country, tgt_list in targets_by_country.items():
        ntf = name_token_freqs[country]
        atf = addr_token_freqs[country]

        for r in tgt_list:
            eid = r.entity_id
            core_toks = strip_legal_suffixes(r.name_tokens)
            core_name = " ".join(core_toks)
            core_sorted = " ".join(sorted(core_toks)) if len(core_toks) > 1 else core_name

            # 1. all_rare_df50
            for t in set(r.name_tokens):
                f = ntf.get(t, 0)
                if 1 <= f <= 50 and len(exp_indices["all_rare_df50"][country][t]) < EXP_KEY_CAP:
                    exp_indices["all_rare_df50"][country][t].append(eid)

            # 2. core_name_exact
            if core_name and core_name != r.name_norm:
                if len(exp_indices["core_name_exact"][country][core_name]) < EXP_KEY_CAP:
                    exp_indices["core_name_exact"][country][core_name].append(eid)

            # 3. core_name_sorted
            if core_sorted and core_sorted != r.name_sorted and core_sorted != core_name:
                if len(exp_indices["core_name_sorted"][country][core_sorted]) < EXP_KEY_CAP:
                    exp_indices["core_name_sorted"][country][core_sorted].append(eid)

            # 4. prefix_3_any_num
            if len(r.name_norm) >= 3 and r.addr_numbers:
                p3 = r.name_norm[:3]
                for num in r.addr_numbers[:2]:
                    k = f"{p3}_{num}"
                    if len(exp_indices["prefix_3_any_num"][country][k]) < EXP_KEY_CAP:
                        exp_indices["prefix_3_any_num"][country][k].append(eid)

            # 5. char_3_last_2
            if len(core_name) >= 6:
                k = f"{core_name[:3]}_{core_name[-2:]}"
                if len(exp_indices["char_3_last_2"][country][k]) < EXP_KEY_CAP:
                    exp_indices["char_3_last_2"][country][k].append(eid)

            # 6. rare_tokens_df100
            for t in set(r.name_tokens):
                f = ntf.get(t, 0)
                if 1 <= f <= 100 and len(exp_indices["rare_tokens_df100"][country][t]) < EXP_KEY_CAP:
                    exp_indices["rare_tokens_df100"][country][t].append(eid)

            # 7. addr_token_num
            if r.addr_tokens and r.addr_numbers:
                best_at = None
                best_af = 999999
                for at in r.addr_tokens:
                    af = atf.get(at, 0)
                    if 1 <= af <= 50 and af < best_af:
                        best_af = af
                        best_at = at
                if best_at:
                    k = f"{best_at}_{r.addr_numbers[0]}"
                    if len(exp_indices["addr_token_num"][country][k]) < EXP_KEY_CAP:
                        exp_indices["addr_token_num"][country][k].append(eid)

            # 8. core_prefix_num
            if core_toks and r.addr_numbers:
                for num in r.addr_numbers[:2]:
                    k = f"{core_toks[0]}_{num}"
                    if len(exp_indices["core_prefix_num"][country][k]) < EXP_KEY_CAP:
                        exp_indices["core_prefix_num"][country][k].append(eid)

    # Function to evaluate an experimental addition to baseline
    def evaluate_blocker_addition(
        name: str,
        query_fn,
        per_s1_cap: int = 50,
    ) -> Dict[str, Any]:
        t_start = time.time()
        new_covered = 0
        incremental_recovered = 0
        cands_per_s1_list = []
        all_new_cands: Dict[str, Set[str]] = {}

        for s1_id in val_s1_ids:
            s1 = s1_all[s1_id]
            country = s1.country
            true_tgts = val_gt.get(s1_id, set())
            base_set = baseline_cands_by_s1[s1_id]

            # Query experimental route
            add_cands = query_fn(s1, country)
            
            # Combine baseline + added
            # Note: priority to baseline routes, then added
            combined = set(base_set) | set(add_cands)
            
            # Cap candidates if exceeding per_s1_cap
            if len(combined) > per_s1_cap:
                # Keep all base_set first, fill rest with add_cands up to per_s1_cap
                combined_list = list(base_set)
                for c in add_cands:
                    if len(combined_list) >= per_s1_cap:
                        break
                    if c not in base_set:
                        combined_list.append(c)
                final_set = set(combined_list)
            else:
                final_set = combined

            all_new_cands[s1_id] = final_set
            cands_per_s1_list.append(len(final_set))

            hit_set = final_set & true_tgts
            new_covered += len(hit_set)
            
            # Incremental hits not in baseline
            inc_hits = hit_set - base_set
            incremental_recovered += len(inc_hits)

        elapsed = time.time() - t_start
        cands_arr = np.array(cands_per_s1_list, dtype=np.int32)
        total_cands = int(np.sum(cands_arr))
        baseline_total_cands = sum(len(baseline_cands_by_s1[s]) for s in val_s1_ids)

        recall = new_covered / total_true_links
        inc_recall = incremental_recovered / total_true_links

        return {
            "name": name,
            "new_true_links_recovered": incremental_recovered,
            "total_covered": new_covered,
            "candidate_recall": float(recall),
            "incremental_recall": float(inc_recall),
            "avg_candidates_per_s1": float(np.mean(cands_arr)),
            "median_candidates_per_s1": float(np.median(cands_arr)),
            "p90": float(np.percentile(cands_arr, 90)),
            "p95": float(np.percentile(cands_arr, 95)),
            "p99": float(np.percentile(cands_arr, 99)),
            "max": int(np.max(cands_arr)),
            "total_candidates": total_cands,
            "candidate_volume_increase_pct": float((total_cands - baseline_total_cands) / baseline_total_cands * 100),
            "reduction_ratio": float(1.0 - (total_cands / (len(val_s1_ids) * len(all_targets)))),
            "runtime_s": float(elapsed),
            "cands_by_s1": all_new_cands,
        }

    # Query functions for individual routes
    def q_all_rare_df50(s1: NormalizedRecord, country: str) -> List[str]:
        idx = exp_indices["all_rare_df50"].get(country, {})
        res = []
        for t in s1.name_tokens:
            if t not in COMMON_STOP_TOKENS and t in idx:
                res.extend(idx[t])
        return res

    def q_core_name_exact(s1: NormalizedRecord, country: str) -> List[str]:
        core_toks = strip_legal_suffixes(s1.name_tokens)
        cn = " ".join(core_toks)
        return exp_indices["core_name_exact"].get(country, {}).get(cn, [])

    def q_core_name_sorted(s1: NormalizedRecord, country: str) -> List[str]:
        core_toks = strip_legal_suffixes(s1.name_tokens)
        cs = " ".join(sorted(core_toks)) if len(core_toks) > 1 else " ".join(core_toks)
        return exp_indices["core_name_sorted"].get(country, {}).get(cs, [])

    def q_prefix_3_any_num(s1: NormalizedRecord, country: str) -> List[str]:
        res = []
        if len(s1.name_norm) >= 3 and s1.addr_numbers:
            p3 = s1.name_norm[:3]
            idx = exp_indices["prefix_3_any_num"].get(country, {})
            for num in s1.addr_numbers[:2]:
                k = f"{p3}_{num}"
                if k in idx:
                    res.extend(idx[k])
        return res

    def q_char_3_last_2(s1: NormalizedRecord, country: str) -> List[str]:
        core_toks = strip_legal_suffixes(s1.name_tokens)
        cn = " ".join(core_toks)
        if len(cn) >= 6:
            k = f"{cn[:3]}_{cn[-2:]}"
            return exp_indices["char_3_last_2"].get(country, {}).get(k, [])
        return []

    def q_rare_tokens_df100(s1: NormalizedRecord, country: str) -> List[str]:
        idx = exp_indices["rare_tokens_df100"].get(country, {})
        res = []
        for t in s1.name_tokens:
            if t not in COMMON_STOP_TOKENS and t in idx:
                res.extend(idx[t])
        return res

    def q_addr_token_num(s1: NormalizedRecord, country: str) -> List[str]:
        res = []
        idx = exp_indices["addr_token_num"].get(country, {})
        if s1.addr_tokens and s1.addr_numbers:
            for at in s1.addr_tokens:
                k = f"{at}_{s1.addr_numbers[0]}"
                if k in idx:
                    res.extend(idx[k])
        return res

    def q_core_prefix_num(s1: NormalizedRecord, country: str) -> List[str]:
        res = []
        core_toks = strip_legal_suffixes(s1.name_tokens)
        if core_toks and s1.addr_numbers:
            idx = exp_indices["core_prefix_num"].get(country, {})
            for num in s1.addr_numbers[:2]:
                k = f"{core_toks[0]}_{num}"
                if k in idx:
                    res.extend(idx[k])
        return res

    # Run evaluations on candidate blockers individually
    experiments = [
        ("1. all_rare_tokens_df50", q_all_rare_df50),
        ("2. core_name_exact", q_core_name_exact),
        ("3. core_name_sorted", q_core_name_sorted),
        ("4. prefix_3_any_num", q_prefix_3_any_num),
        ("5. char_3_last_2", q_char_3_last_2),
        ("6. rare_tokens_df100", q_rare_tokens_df100),
        ("7. addr_token_num", q_addr_token_num),
        ("8. core_prefix_num", q_core_prefix_num),
    ]

    exp_results = []
    logger.info("============================================================")
    logger.info("EVALUATING INDIVIDUAL CANDIDATE BLOCKERS (cap=50/S1)")
    logger.info("============================================================")

    for name, q_fn in experiments:
        res = evaluate_blocker_addition(name, q_fn, per_s1_cap=50)
        exp_results.append(res)
        logger.info(
            f"{res['name']:25s} | Recov: +{res['new_true_links_recovered']:4d} | "
            f"Recall: {res['candidate_recall']*100:5.2f}% (+{res['incremental_recall']*100:4.2f}%) | "
            f"Avg/S1: {res['avg_candidates_per_s1']:5.2f} | P95: {res['p95']:.0f} | P99: {res['p99']:.0f} | "
            f"Vol: +{res['candidate_volume_increase_pct']:5.1f}% | Time: {res['runtime_s']:.2f}s"
        )

    # Test with higher S1 cap (e.g. cap=60 or 75) for top combinations
    logger.info("============================================================")
    logger.info("TESTING COMPOSITE BLOCKER CONFIGURATIONS")
    logger.info("============================================================")

    # Composite Combo A: Baseline + all_rare_df50 + core_name_exact + core_name_sorted
    def q_combo_a(s1: NormalizedRecord, country: str) -> List[str]:
        r1 = q_all_rare_df50(s1, country)
        r2 = q_core_name_exact(s1, country)
        r3 = q_core_name_sorted(s1, country)
        return r1 + r2 + r3

    # Composite Combo B: Combo A + prefix_3_any_num + core_prefix_num
    def q_combo_b(s1: NormalizedRecord, country: str) -> List[str]:
        r1 = q_combo_a(s1, country)
        r4 = q_prefix_3_any_num(s1, country)
        r5 = q_core_prefix_num(s1, country)
        return r1 + r4 + r5

    # Composite Combo C: Combo B + rare_tokens_df100
    def q_combo_c(s1: NormalizedRecord, country: str) -> List[str]:
        r1 = q_combo_b(s1, country)
        r6 = q_rare_tokens_df100(s1, country)
        return r1 + r6

    composites = [
        ("Combo A (core_names + all_rare50)", q_combo_a, 50),
        ("Combo A (core_names + all_rare50) [cap=65]", q_combo_a, 65),
        ("Combo B (Combo A + prefix3_num + core_prefix_num)", q_combo_b, 50),
        ("Combo B (Combo A + prefix3_num + core_prefix_num) [cap=65]", q_combo_b, 65),
        ("Combo B [cap=75]", q_combo_b, 75),
        ("Combo C (Combo B + rare100) [cap=65]", q_combo_c, 65),
        ("Combo C (Combo B + rare100) [cap=75]", q_combo_c, 75),
    ]

    composite_results = []
    for name, q_fn, cap in composites:
        res = evaluate_blocker_addition(name, q_fn, per_s1_cap=cap)
        composite_results.append(res)
        logger.info(
            f"{res['name']:48s} | Recov: +{res['new_true_links_recovered']:4d} | "
            f"Recall: {res['candidate_recall']*100:5.2f}% (+{res['incremental_recall']*100:4.2f}%) | "
            f"Avg/S1: {res['avg_candidates_per_s1']:5.2f} | P95: {res['p95']:.0f} | P99: {res['p99']:.0f} | "
            f"Vol: +{res['candidate_volume_increase_pct']:5.1f}% | Time: {res['runtime_s']:.2f}s"
        )

    # Save results to JSON
    # Strip cands_by_s1 for compact JSON
    clean_exp_res = [{k: v for k, v in r.items() if k != "cands_by_s1"} for r in exp_results]
    clean_comp_res = [{k: v for k, v in r.items() if k != "cands_by_s1"} for r in composite_results]

    exp_report = {
        "baseline": {
            "total_true_links": total_true_links,
            "covered": base_covered,
            "candidate_recall": float(base_covered / total_true_links),
            "avg_candidates_per_s1": float(sum(len(baseline_cands_by_s1[s]) for s in val_s1_ids) / len(val_s1_ids)),
            "runtime_s": baseline_time,
        },
        "individual_blockers": clean_exp_res,
        "composite_blockers": clean_comp_res,
    }

    with open(reports_dir / "blocker_experiments.json", "w", encoding="utf-8") as f:
        json.dump(exp_report, f, indent=2)

    logger.info("Saved blocker experiments report to artifacts/reports/blocker_experiments.json")


if __name__ == "__main__":
    main()
