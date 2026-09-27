import csv
import gc
import sys
import time
from collections import defaultdict

class CompactRecord:
    __slots__ = ("entity_id", "name_norm", "name_sorted", "addr_norm", "addr_missing", "name_tokens", "addr_tokens", "addr_numbers")
    def __init__(self, eid, name_norm, name_sorted, addr_norm, addr_missing, name_tokens, addr_tokens, addr_numbers):
        self.entity_id = eid
        self.name_norm = name_norm
        self.name_sorted = name_sorted
        self.addr_norm = addr_norm
        self.addr_missing = addr_missing
        self.name_tokens = name_tokens
        self.addr_tokens = addr_tokens
        self.addr_numbers = addr_numbers

import os
sys.path.insert(0, os.getcwd())
from main import normalize_text, tokenize, extract_numbers

def run_test():
    t0 = time.time()
    targets = []
    print("Loading 500,000 India targets...")
    with open("data/raw/test/test_source2.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            row = line.rstrip("\r\n").split("\t")
            if len(row) >= 4 and row[3].strip().upper() == "INDIA":
                eid = sys.intern(row[0])
                n_norm = normalize_text(row[1])
                n_toks = tuple(sys.intern(t) for t in tokenize(n_norm))
                n_sorted = " ".join(sorted(n_toks)) if len(n_toks) > 1 else n_norm
                a_raw = row[2].strip()
                a_miss = (not a_raw or a_raw.lower() in ("null", "none", "nan", ""))
                a_norm = normalize_text(a_raw) if not a_miss else ""
                a_toks = tuple(sys.intern(t) for t in tokenize(a_norm)) if not a_miss else ()
                a_nums = tuple(sys.intern(t) for t in extract_numbers(a_norm)) if not a_miss else ()
                targets.append(CompactRecord(eid, n_norm, n_sorted, a_norm, a_miss, n_toks, a_toks, a_nums))
                if len(targets) >= 500000:
                    break

    print(f"Loaded 500,000 targets in {time.time()-t0:.2f}s")
    from scratch.check_mem import check
    check()

if __name__ == "__main__":
    run_test()
