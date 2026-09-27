import csv
import sys
import time

def fast_validate():
    t0 = time.time()
    s1_path = "data/raw/test/test_source1.tsv"
    m_path = "output/matching_results.tsv"
    c_path = "output/candidate_pairs.tsv"

    print("Fast Streaming Submission Validator")
    print(f"Checking {m_path} and {c_path} against {s1_path}...")

    with open(s1_path, "r", encoding="utf-8") as f_s1, \
         open(m_path, "r", encoding="utf-8") as f_m, \
         open(c_path, "r", encoding="utf-8") as f_c:

        # Check headers
        m_head = f_m.readline().rstrip("\r\n").split("\t")
        c_head = f_c.readline().rstrip("\r\n").split("\t")
        next(f_s1) # skip s1 header

        assert m_head == ["source1_entity_id", "matched_entity_ids"], f"Invalid matching header: {m_head}"
        assert c_head == ["source1_entity_id", "candidate_entity_ids"], f"Invalid candidate header: {c_head}"

        row_count = 0
        total_matches = 0
        total_candidates = 0
        max_matches_found = 0
        zero_matches = 0
        single_matches = 0

        for line_num, (l_s1, l_m, l_c) in enumerate(zip(f_s1, f_m, f_c), start=2):
            r_s1 = l_s1.rstrip("\r\n").split("\t")
            r_m = l_m.rstrip("\r\n").split("\t")
            r_c = l_c.rstrip("\r\n").split("\t")

            s1_id = r_s1[0].strip()
            m_s1_id = r_m[0].strip()
            c_s1_id = r_c[0].strip()

            if m_s1_id != s1_id:
                raise ValueError(f"Line {line_num}: Matching ID mismatch! Expected {s1_id}, got {m_s1_id}")
            if c_s1_id != s1_id:
                raise ValueError(f"Line {line_num}: Candidate ID mismatch! Expected {s1_id}, got {c_s1_id}")

            m_ids = r_m[1].split(",") if len(r_m) > 1 and r_m[1].strip() else []
            c_ids = r_c[1].split(",") if len(r_c) > 1 and r_c[1].strip() else []

            # Check max matches <= 15
            if len(m_ids) > 15:
                raise ValueError(f"Line {line_num}: {s1_id} has {len(m_ids)} matches > 15!")

            # Check no duplicates within lists
            if len(m_ids) != len(set(m_ids)):
                raise ValueError(f"Line {line_num}: {s1_id} has duplicate matches!")
            if len(c_ids) != len(set(c_ids)):
                raise ValueError(f"Line {line_num}: {s1_id} has duplicate candidates!")

            # Check candidate superset: all matches must be in candidates
            c_set = set(c_ids)
            for mid in m_ids:
                if mid not in c_set:
                    raise ValueError(f"Line {line_num}: Matched ID {mid} not in candidate set for {s1_id}!")
                if not (mid.startswith("S2-") or mid.startswith("S3-")):
                    raise ValueError(f"Line {line_num}: Invalid target ID prefix: {mid}")

            row_count += 1
            n_m = len(m_ids)
            total_matches += n_m
            total_candidates += len(c_ids)
            if n_m > max_matches_found:
                max_matches_found = n_m
            if n_m == 0:
                zero_matches += 1
            elif n_m == 1:
                single_matches += 1

            if row_count % 500000 == 0:
                print(f"Validated {row_count:,} / 1,732,544 rows...")

        # Ensure no trailing lines in any file
        assert f_m.readline() == "", "Extra lines in matching_results.tsv!"
        assert f_c.readline() == "", "Extra lines in candidate_pairs.tsv!"
        assert f_s1.readline() == "", "Extra lines in test_source1.tsv!"

    print(f"\nSUCCESS! 100% VALIDATION PASSED in {time.time()-t0:.2f}s!")
    print(f"Total Rows: {row_count:,} (matches test_source1.tsv exactly)")
    print(f"Total Matches: {total_matches:,} (mean: {total_matches/row_count:.2f}/S1)")
    print(f"Total Candidates: {total_candidates:,} (mean: {total_candidates/row_count:.2f}/S1)")
    print(f"Max Matches / S1: {max_matches_found} (<= 15 constraint strictly satisfied)")
    print(f"Zero-Match S1: {zero_matches:,} ({zero_matches/row_count*100:.1f}%)")
    print(f"Single-Match S1: {single_matches:,} ({single_matches/row_count*100:.1f}%)")
    print(f"Multi-Match S1: {row_count - zero_matches - single_matches:,} ({(row_count - zero_matches - single_matches)/row_count*100:.1f}%)")
    print("Candidate Superset Constraint: 100% VERIFIED (0 violations)")

if __name__ == "__main__":
    fast_validate()
