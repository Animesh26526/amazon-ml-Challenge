import json
from pathlib import Path

with open("artifacts/reports/blocking_decomposition.json") as f:
    dec = json.load(f)

with open("artifacts/reports/blocker_experiments.json") as f:
    exp = json.load(f)

with open("artifacts/reports/advanced_blocking_evaluation.json") as f:
    adv = json.load(f)

print("=== EXISTING BLOCKER STATS ===")
for k, v in dec["block_size_stats"].items():
    print(f"{k:15s}: Keys={v['total_keys']:,}, P50={v['p50']}, P90={v['p90']}, P95={v['p95']}, P99={v['p99']}, P99.9={v['p99.9']}, Max={v['max']}, TruncatedKeys={v['keys_truncated']} ({v['pct_keys_truncated']:.2f}%)")

print("\n=== ROUTE TRUE HITS IN BASELINE ===")
for k, v in dec["route_true_hits"].items():
    print(f"  {k:15s}: {v:,}")

print("\n=== BASELINE TRUNCATION LOSSES ===")
print(f"  Loss to S1 cap (50): {dec['loss_to_s1_cap']:,}")
print(f"  Loss to Key cap (60): {dec['loss_to_key_trunc']:,}")

print("\n=== FAILURE CATEGORIES (out of 10,339 misses) ===")
miss_count = dec["missed_links_baseline"]
for k, v in sorted(dec["category_counts"].items(), key=lambda x: x[1], reverse=True):
    print(f"  {k:35s}: {v:5,d} ({v/miss_count*100:5.2f}%)")

print("\n=== RETRIEVABILITY ANALYSIS ===")
for k, v in sorted(dec["retrievability_counts"].items(), key=lambda x: x[1], reverse=True):
    print(f"  {k:30s}: {v:5,d} ({v/miss_count*100:5.2f}%)")

print("\n=== NEW BLOCKER BLOCK SIZES ===")
for k, v in adv["new_blocker_block_sizes"].items():
    print(f"{k:20s}: Keys={v['total_keys']:,}, P50={v['p50']}, P90={v['p90']}, P95={v['p95']}, P99={v['p99']}, P99.9={v['p99.9']}, Max={v['max']}, TruncatedKeys={v['keys_truncated']} ({v['pct_keys_truncated']:.2f}%)")

print("\n=== INDIVIDUAL BLOCKER EXPERIMENTS ===")
for r in exp["individual_blockers"]:
    print(f"{r['name']:25s} | Recov: +{r['new_true_links_recovered']:4d} | Rec: {r['candidate_recall']*100:5.2f}% (+{r['incremental_recall']*100:4.2f}%) | Avg/S1: {r['avg_candidates_per_s1']:5.2f} | P90: {r['p90']:.0f} | P95: {r['p95']:.0f} | P99: {r['p99']:.0f} | Max: {r['max']} | Vol: +{r['candidate_volume_increase_pct']:5.1f}% | Time: {r['runtime_s']:.2f}s")

print("\n=== END-TO-END PIPELINE EVALUATION ===")
for r in adv["evaluations"]:
    std = r["standard_metrics"]
    rel = r["relief_metrics"]
    print(f"{r['name']:42s} | Rec: {r['candidate_recall']*100:5.2f}% | Avg: {r['avg_candidates_per_s1']:5.2f} | P95: {r['p95']:.0f} | P99: {r['p99']:.0f} | Macro F0.5: {std['macro_f05']:.4f} | Prec: {std['precision']:.4f} | Rec: {std['recall']:.4f} (Relief F0.5: {rel['macro_f05']:.4f})")
