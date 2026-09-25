"""Input/Output and Schema Utilities.

Provides high-performance, memory-safe data loading, schema validation,
metadata inspection, and strict output TSV writers adhering to the
competition submission protocol and invariants.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Expected canonical column definitions
SOURCE_COLUMNS: List[str] = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_COLUMNS: List[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

MATCHING_RESULTS_COLUMNS: List[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

CANDIDATE_PAIRS_COLUMNS: List[str] = [
    "source1_entity_id",
    "candidate_entity_ids",
]


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Nested dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file is not valid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing configuration YAML {path}: {exc}") from exc

    return cfg or {}


def parse_id_list(val: Any) -> List[str]:
    """Parse a comma-separated ID string into a list of clean IDs.

    Args:
        val: String containing comma-separated IDs, or float/None if missing.

    Returns:
        List of trimmed ID strings (empty list if missing or empty).
    """
    if val is None or pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def format_id_list(ids: Iterable[str]) -> str:
    """Format an iterable of IDs into a canonical comma-separated string.

    Deduplicates IDs while preserving order.

    Args:
        ids: Iterable of entity ID strings.

    Returns:
        Comma-separated ID string with no surrounding whitespace.
    """
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in ids:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)
    return ",".join(ordered)


def read_source_tsv(
    path: Union[str, Path],
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
    usecols: Optional[Sequence[str]] = None,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Read a source TSV file (S1, S2, or S3) with schema verification.

    Args:
        path: Path to the TSV file.
        nrows: Optional number of rows to read.
        chunksize: Optional chunk size for streaming iteration.
        usecols: Optional subset of columns to load.

    Returns:
        A DataFrame or an iterator of DataFrame chunks.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain expected columns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    target_cols = list(usecols) if usecols is not None else SOURCE_COLUMNS
    dtype = {
        "entity_id": str,
        "business_name": str,
        "business_address": str,
        "country": str,
    }

    return pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        chunksize=chunksize,
        usecols=target_cols,
        dtype=dtype,
        keep_default_na=False,
        encoding="utf-8",
    )


def read_ground_truth_tsv(
    path: Union[str, Path],
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Read train_ground_truth.tsv into a standardized DataFrame.

    Args:
        path: Path to ground truth TSV file.
        nrows: Optional number of rows to load.

    Returns:
        DataFrame with columns ['source1_entity_id', 'matched_entity_ids'].
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    return df


def inspect_file_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect lightweight metadata of a file without loading it completely.

    Args:
        path: Path to the target file.

    Returns:
        Dictionary with 'path', 'size_bytes', 'size_mb', 'header', and 'estimated_rows'.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    size_bytes = path.stat().st_size
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\r\n")

    cols = first_line.split("\t") if "\t" in first_line else first_line.split(",")
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "header": cols,
        "delimiter": "\t" if "\t" in first_line else ",",
    }


def write_submission_tsv(
    records: Union[pd.DataFrame, Dict[str, Sequence[str]]],
    output_path: Union[str, Path],
    id_column: str,
    list_column: str,
) -> None:
    """Write submission-compliant TSV file.

    Ensures no quoting, exact tab separator, comma-separated ID lists,
    and trailing newline.

    Args:
        records: Mapping from S1 ID to sequence of target IDs, or DataFrame.
        output_path: Destination TSV file path.
        id_column: Name of first column (e.g. 'source1_entity_id').
        list_column: Name of second column (e.g. 'matched_entity_ids').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            lineterminator="\n",
            escapechar="\\",
        )
        writer.writerow([id_column, list_column])

        if isinstance(records, pd.DataFrame):
            for _, row in records.iterrows():
                s1_id = str(row[id_column]).strip()
                val = row[list_column]
                if isinstance(val, (list, set, tuple)):
                    formatted = format_id_list(val)
                else:
                    formatted = format_id_list(parse_id_list(val))
                writer.writerow([s1_id, formatted])
        elif isinstance(records, dict):
            for s1_id, target_ids in records.items():
                s1_clean = str(s1_id).strip()
                formatted = format_id_list(target_ids)
                writer.writerow([s1_clean, formatted])
        else:
            raise TypeError(f"Unsupported records type: {type(records)}")


def write_matching_results(
    records: Union[pd.DataFrame, Dict[str, Sequence[str]]],
    output_path: Union[str, Path],
) -> None:
    """Write matching_results.tsv adhering strictly to competition format."""
    write_submission_tsv(
        records=records,
        output_path=output_path,
        id_column="source1_entity_id",
        list_column="matched_entity_ids",
    )


def write_candidate_pairs(
    records: Union[pd.DataFrame, Dict[str, Sequence[str]]],
    output_path: Union[str, Path],
) -> None:
    """Write candidate_pairs.tsv adhering strictly to competition format."""
    write_submission_tsv(
        records=records,
        output_path=output_path,
        id_column="source1_entity_id",
        list_column="candidate_entity_ids",
    )


def verify_submission_invariants(
    matching_path: Union[str, Path],
    candidate_path: Optional[Union[str, Path]] = None,
    test_source1_path: Optional[Union[str, Path]] = None,
) -> Tuple[bool, List[str]]:
    """Verify internal pipeline invariants on generated submission TSVs.

    Checks:
      1. Correct TSV header.
      2. No duplicate S1 rows.
      3. No duplicate IDs inside match/candidate lists.
      4. No self-matches (S1- prefix in match/candidate lists).
      5. Only valid prefixes (S2-, S3-).
      6. Every matched ID must be a candidate ID: matched_set <= candidate_set.
      7. (Optional) All required test S1 entities are present.

    Args:
        matching_path: Path to matching_results.tsv.
        candidate_path: Optional path to candidate_pairs.tsv.
        test_source1_path: Optional path to test_source1.tsv.

    Returns:
        (is_valid, error_list)
    """
    errors: List[str] = []
    matching_path = Path(matching_path)
    if not matching_path.is_file():
        return False, [f"Matching file not found: {matching_path}"]

    matches_map: Dict[str, Set[str]] = {}
    with matching_path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header != MATCHING_RESULTS_COLUMNS:
            errors.append(f"Invalid matching header: {header}, expected {MATCHING_RESULTS_COLUMNS}")

        for line_idx, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                errors.append(f"Malformed row at line {line_idx} in {matching_path.name}")
                continue
            s1_id, match_str = parts[0].strip(), parts[1].strip()
            if s1_id in matches_map:
                errors.append(f"Duplicate S1 ID in matching: {s1_id} at line {line_idx}")
            ids = parse_id_list(match_str)
            if len(ids) != len(set(ids)):
                errors.append(f"Duplicate IDs inside matched list for {s1_id}")
            for m in ids:
                if m.startswith("S1-"):
                    errors.append(f"Self-match detected: {m} in {s1_id}")
                elif not (m.startswith("S2-") or m.startswith("S3-")):
                    errors.append(f"Invalid prefix: {m} in {s1_id}")
            matches_map[s1_id] = set(ids)

    # Cross-check candidates if provided
    if candidate_path is not None:
        candidate_path = Path(candidate_path)
        if not candidate_path.is_file():
            errors.append(f"Candidate file not found: {candidate_path}")
        else:
            candidates_map: Dict[str, Set[str]] = {}
            with candidate_path.open("r", encoding="utf-8") as f:
                c_header = f.readline().rstrip("\r\n").split("\t")
                if c_header != CANDIDATE_PAIRS_COLUMNS:
                    errors.append(f"Invalid candidate header: {c_header}")

                for line_idx, line in enumerate(f, start=2):
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) != 2:
                        errors.append(f"Malformed candidate row at line {line_idx}")
                        continue
                    s1_id, cand_str = parts[0].strip(), parts[1].strip()
                    if s1_id in candidates_map:
                        errors.append(f"Duplicate S1 ID in candidates: {s1_id} at line {line_idx}")
                    c_ids = parse_id_list(cand_str)
                    candidates_map[s1_id] = set(c_ids)

            # Invariant: predicted_matches <= candidate_pairs
            for s1_id, matched_set in matches_map.items():
                cand_set = candidates_map.get(s1_id, set())
                uncovered = matched_set - cand_set
                if uncovered:
                    sample = list(uncovered)[:3]
                    errors.append(
                        f"Candidate invariant violated for {s1_id}: "
                        f"{len(uncovered)} predicted matches not in candidate set, e.g. {sample}"
                    )

    # Optional test S1 coverage check
    if test_source1_path is not None:
        test_path = Path(test_source1_path)
        if test_path.is_file():
            test_s1_ids: Set[str] = set()
            with test_path.open("r", encoding="utf-8") as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.split("\t", 1)
                    if parts:
                        test_s1_ids.add(parts[0].strip())

            missing = test_s1_ids - set(matches_map.keys())
            if missing:
                errors.append(f"Missing {len(missing)} required test S1 entities in matching results")
            extra = set(matches_map.keys()) - test_s1_ids
            if extra:
                errors.append(f"{len(extra)} unexpected S1 entities present in matching results")

    return len(errors) == 0, errors
