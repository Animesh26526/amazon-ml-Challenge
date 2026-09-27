import csv
import gc
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

import os
sys.path.insert(0, os.getcwd())

from main import NormalizedRecord, GBDTMatcher, CandidatePair, COMMON_STOP_TOKENS, format_id_list
from rapidfuzz import fuzz

class CompactIndiaTarget:
    __slots__ = (
        "entity_id", "name_norm", "name_sorted", "addr_norm", "addr_missing",
        "name_tokens", "addr_tokens", "addr_numbers",
        "name_tokens_set", "addr_tokens_set", "addr_numbers_set"
    )
    def __init__(self, eid, name_norm, name_sorted, addr_norm, addr_missing, name_toks, addr_toks, addr_nums):
        self.entity_id = eid
        self.name_norm = name_norm
        self.name_sorted = name_sorted
        self.addr_norm = addr_norm
        self.addr_missing = addr_missing
        self.name_tokens = name_toks
        self.addr_tokens = addr_toks
        self.addr_numbers = addr_nums
        self.name_tokens_set = set(name_toks)
        self.addr_tokens_set = set(addr_toks)
        self.addr_numbers_set = set(addr_nums)

def fast_extract(s1, tgt, pair, s1_nt, s1_at, s1_nums):
    name_exact = 1.0 if (s1.name_norm and s1.name_norm == tgt.name_norm) else 0.0
    tgt_nt = tgt.name_tokens_set
    if name_exact == 1.0:
        name_jaccard = 1.0; name_overlap = 1.0; name_edit = 1.0; name_sort = 1.0; name_len_diff = 0.0; name_len_ratio = 1.0
    else:
        inter = len(s1_nt & tgt_nt)
        name_jaccard = inter / len(s1_nt | tgt_nt) if (s1_nt or tgt_nt) else 0.0
        name_overlap = inter / min(len(s1_nt), len(tgt_nt)) if (s1_nt and tgt_nt) else 0.0
        name_edit = fuzz.ratio(s1.name_norm, tgt.name_norm) / 100.0 if (s1.name_norm and tgt.name_norm) else 0.0
        name_sort = fuzz.token_sort_ratio(s1.name_norm, tgt.name_norm) / 100.0 if (s1.name_norm and tgt.name_norm) else 0.0
        name_len_diff = float(abs(len(s1.name_norm) - len(tgt.name_norm)))
        max_len = max(len(s1.name_norm), len(tgt.name_norm), 1)
        name_len_ratio = float(min(len(s1.name_norm), len(tgt.name_norm))) / float(max_len)

    addr_exact = 1.0 if (not s1.addr_missing and not tgt.addr_missing and s1.addr_norm == tgt.addr_norm) else 0.0
    tgt_at = tgt.addr_tokens_set
    if addr_exact == 1.0:
        addr_jaccard = 1.0; addr_overlap = 1.0; addr_edit = 1.0
    elif s1.addr_missing or tgt.addr_missing:
        addr_jaccard = 0.0; addr_overlap = 0.0; addr_edit = 0.0
    else:
        inter_a = len(s1_at & tgt_at)
        addr_jaccard = inter_a / len(s1_at | tgt_at) if (s1_at or tgt_at) else 0.0
        addr_overlap = inter_a / min(len(s1_at), len(tgt_at)) if (s1_at and tgt_at) else 0.0
        addr_edit = fuzz.ratio(s1.addr_norm, tgt.addr_norm) / 100.0

    tgt_nums = tgt.addr_numbers_set
    inter_num = len(s1_nums & tgt_nums)
    num_jaccard = inter_num / len(s1_nums | tgt_nums) if (s1_nums or tgt_nums) else 0.0
    num_overlap = inter_num / min(len(s1_nums), len(tgt_nums)) if (s1_nums and tgt_nums) else 0.0

    addr_miss_s1 = 1.0 if s1.addr_missing else 0.0
    addr_miss_tgt = 1.0 if tgt.addr_missing else 0.0
    addr_miss_either = 1.0 if (s1.addr_missing or tgt.addr_missing) else 0.0

    prod_sim = name_edit * addr_edit
    min_sim = min(name_edit, addr_edit)
    mean_sim = (name_edit + addr_edit) / 2.0
    both_high = 1.0 if (name_edit > 0.80 and addr_edit > 0.80) else 0.0
    name_hi_addr_lo = 1.0 if (name_edit > 0.85 and addr_edit < 0.35) else 0.0
    addr_hi_name_lo = 1.0 if (addr_edit > 0.85 and name_edit < 0.35) else 0.0
    name_hi_addr_miss = 1.0 if (name_edit > 0.80 and tgt.addr_missing) else 0.0

    is_s2 = 1.0
    is_s3 = 0.0
    r_exact_n = float(pair.route_exact_name)
    r_sort_n = float(pair.route_sorted_name)
    r_exact_a = float(pair.route_exact_addr)
    r_num_a = float(pair.route_numeric_anchor)
    r_rare_t = float(pair.route_rare_token)
    r_char_n = float(pair.route_char_ngram)
    r_count = float(pair.route_count)
    same_c = 1.0

    return [
        name_exact, name_jaccard, name_overlap, name_edit, name_sort, name_len_diff, name_len_ratio,
        addr_exact, addr_jaccard, addr_overlap, addr_edit, num_jaccard, num_overlap,
        addr_miss_s1, addr_miss_tgt, addr_miss_either,
        prod_sim, min_sim, mean_sim, both_high, name_hi_addr_lo, addr_hi_name_lo, name_hi_addr_miss,
        is_s2, is_s3, r_exact_n, r_sort_n, r_exact_a, r_num_a, r_rare_t, r_char_n, r_count, same_c
    ]

print("Script syntax verified.")
