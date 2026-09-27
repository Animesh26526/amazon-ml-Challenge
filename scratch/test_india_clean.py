import gc
import sys
import time
from pathlib import Path
import os
sys.path.insert(0, os.getcwd())

from main import NormalizedRecord
from tools.run_final_test_inference import RecommendedMultiModalBlocker
from scratch.check_mem import check

def test_india():
    print("Testing clean India S2 load & index in fresh process...")
    t0 = time.time()
    s2_targets = {}
    with open("data/raw/test/test_source2.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            row = line.rstrip("\r\n").split("\t")
            if len(row) >= 4 and row[3].strip().upper() == "INDIA":
                rec = NormalizedRecord.from_raw(row[0], row[1], row[2], row[3], store_raw=False)
                s2_targets[rec.entity_id] = rec

    print(f"Loaded {len(s2_targets):,} India S2 targets in {time.time()-t0:.1f}s")
    check()

    t1 = time.time()
    blocker = RecommendedMultiModalBlocker(max_candidates_per_key=60, max_candidates_per_s1=30)
    blocker.build_indexes(list(s2_targets.values()))
    print(f"Built India S2 blocker in {time.time()-t1:.1f}s")
    check()

if __name__ == "__main__":
    test_india()
