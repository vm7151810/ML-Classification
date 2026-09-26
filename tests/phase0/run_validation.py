"""Automated Validation and Quality Checks for Phase 0.

Implements Tier 1 (structural correctness) and Tier 2 (distributional regression)
checks per phase0_validation_checks.md, plus Tier 3 spot-sample exports.
"""

import glob
import os
import sys
import pandas as pd
import pyarrow.parquet as pq


def validate_split(
    processed_dir: str,
    raw_dataset_dir: str,
    split: str,
    spot_sample_out: str | None = None,
) -> bool:
    """Run Tier 1 and Tier 2 validation checks for a split.

    Returns:
        True if all Tier 1 checks pass and Tier 2 checks pass or warn within tolerance.
    """
    print(f"\n========================================================")
    print(f"RUNNING VALIDATION CHECKS FOR SPLIT: {split.upper()}")
    print(f"========================================================")

    split_dir = os.path.join(processed_dir, split)
    parquet_files = sorted(glob.glob(os.path.join(split_dir, "*.parquet")))

    if not parquet_files:
        print(f"[FAIL] No parquet files found in {split_dir}")
        return False

    print(f"Found {len(parquet_files)} parquet partition files:")
    for pf in parquet_files:
        print(f"  - {os.path.basename(pf)}")

    # ----------------------------------------------------
    # TIER 1: Structural Correctness (Hard Fail)
    # ----------------------------------------------------
    print("\n--- Tier 1: Structural Correctness Checks ---")
    tier1_passed = True

    # 1. Load all partitions into metadata summaries
    partition_dfs = {}
    source_counts = {"S1": 0, "S2": 0, "S3": 0}
    all_entity_ids = {}  # eid -> country
    duplicate_eids_in_file = 0
    cross_country_leakage = 0
    empty_names_faiss = 0
    empty_names_bm25 = 0
    ufffd_detected = 0

    cross_script_sample_rows = []

    for pf in parquet_files:
        filename = os.path.basename(pf)
        # Parse role and country from filename e.g. Query_US.parquet
        parts = filename.replace(".parquet", "").split("_")
        role = parts[0]
        country = parts[1]

        # Read parquet columns
        table = pq.read_table(pf)
        df = table.to_pandas()
        partition_dfs[filename] = df
        nrows = len(df)
        print(f"Partition {filename}: {nrows:,} rows")

        # Uniqueness of entity_id within file
        unique_eids = df["entity_id"].nunique()
        if unique_eids != nrows:
            print(f"[FAIL] {filename} has {nrows - unique_eids} duplicate entity_ids!")
            tier1_passed = False
            duplicate_eids_in_file += (nrows - unique_eids)

        # Cross-country leakage check
        for eid in df["entity_id"]:
            if eid in all_entity_ids:
                if all_entity_ids[eid] != country:
                    print(f"[FAIL] Entity {eid} appears in both {all_entity_ids[eid]} and {country}!")
                    cross_country_leakage += 1
                    tier1_passed = False
            else:
                all_entity_ids[eid] = country

        # Source breakdown
        for s_prefix, cnt in df["source"].value_counts().items():
            source_counts[s_prefix] += cnt

        # Check for empty name_for_faiss or name_for_bm25
        empty_f = (df["name_for_faiss"].isna() | (df["name_for_faiss"].str.strip() == "")).sum()
        empty_b = (df["name_for_bm25"].isna() | (df["name_for_bm25"].str.strip() == "")).sum()
        if empty_f > 0:
            print(f"[FAIL] {filename} has {empty_f} empty name_for_faiss!")
            empty_names_faiss += empty_f
            tier1_passed = False
        if empty_b > 0:
            print(f"[FAIL] {filename} has {empty_b} empty name_for_bm25!")
            empty_names_bm25 += empty_b
            tier1_passed = False

        # Check for replacement character U+FFFD in names
        ufffd_f = df["name_for_faiss"].str.contains("\ufffd", na=False).sum()
        ufffd_b = df["name_for_bm25"].str.contains("\ufffd", na=False).sum()
        if ufffd_f > 0 or ufffd_b > 0:
            print(f"[FAIL] {filename} contains U+FFFD replacement characters!")
            ufffd_detected += (ufffd_f + ufffd_b)
            tier1_passed = False

        # Check for untransliterated native-script or non-ASCII characters leaking into name_for_bm25
        indic_scripts = {"Devanagari", "Tamil", "Telugu", "Kannada", "Bengali", "Gujarati", "Gurmukhi", "Malayalam", "Oriya"}
        indic_cs = df[(df["is_cross_script"] == True) & (df["script"].isin(indic_scripts))]
        if not indic_cs.empty:
            leaked_non_ascii = indic_cs["name_for_bm25"].apply(lambda s: any(ord(c) > 127 for c in str(s))).sum()
            if leaked_non_ascii > 0:
                print(f"[FAIL] {filename} has {leaked_non_ascii:,} transliterated rows with leaked non-ASCII characters!")
                tier1_passed = False
            else:
                print(f"  [PASS] {filename}: 0 leaked non-ASCII characters in {len(indic_cs):,} transliterated rows")

        # Collect cross-script rows for Tier 3 spot sample
        cs_rows = df[df["is_cross_script"] == True][["entity_id", "country", "script", "name_original", "name_for_faiss", "name_for_bm25"]]
        if not cs_rows.empty:
            cross_script_sample_rows.append(cs_rows.head(25))

    # Check total input vs output row counts per source file
    raw_dir = os.path.join(raw_dataset_dir, split)
    expected_files = [
        (f"{split}_source1.tsv", "S1"),
        (f"{split}_source2.tsv", "S2"),
        (f"{split}_source3.tsv", "S3"),
    ]

    for fname, prefix in expected_files:
        fpath = os.path.join(raw_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                # count lines minus header
                raw_count = sum(1 for _ in f) - 1
            out_count = source_counts[prefix]
            print(f"Source {prefix} ({fname}): raw={raw_count:,}, processed={out_count:,}")
            if raw_count != out_count:
                print(f"[FAIL] Row count mismatch for {fname}: raw {raw_count} != output {out_count}!")
                tier1_passed = False
            else:
                print(f"  [PASS] Row count matches exactly ({raw_count:,})")

    if tier1_passed:
        print("[PASS] All Tier 1 Structural Checks PASSED cleanly!")
    else:
        print("[FAIL] One or more Tier 1 Structural Checks FAILED!")

    # ----------------------------------------------------
    # TIER 2: Distributional Regression Checks
    # ----------------------------------------------------
    print("\n--- Tier 2: Distributional Regression Checks ---")
    tier2_warnings = 0

    # Combine partitions for India to check cross-script & address missingness rates
    india_dfs = [df for fn, df in partition_dfs.items() if "_India.parquet" in fn]
    if india_dfs:
        df_india = pd.concat(india_dfs, ignore_index=True)
        for s in ["S1", "S2", "S3"]:
            sub = df_india[df_india["source"] == s]
            if len(sub) > 0:
                cs_rate = (sub["is_cross_script"] == True).mean()
                addr_missing = (sub["has_address"] == False).mean()
                print(f"India {s}: rows={len(sub):,}, is_cross_script={cs_rate:.2%}, missing_address={addr_missing:.2%}")

                # Baselines from EDA:
                # Total S2 non-Latin: ~9.1% across all S2 (which is ~22-24% of India S2)
                # Total S3 non-Latin: ~5.0% across all S3 (which is ~11-13% of India S3)
                if s == "S1":
                    if cs_rate > 0.01:
                        print(f"  [WARN] S1 cross-script rate {cs_rate:.2%} higher than expected ~0%")
                        tier2_warnings += 1
                elif s == "S2":
                    if not (0.18 <= cs_rate <= 0.28):
                        print(f"  [WARN] India S2 cross-script rate {cs_rate:.2%} outside ~20-25% India baseline (~9.1% overall)")
                        tier2_warnings += 1
                    if not (0.02 <= addr_missing <= 0.05):
                        print(f"  [WARN] India S2 missing address rate {addr_missing:.2%} outside ~3.3% tolerance")
                        tier2_warnings += 1
                elif s == "S3":
                    if not (0.09 <= cs_rate <= 0.16):
                        print(f"  [WARN] India S3 cross-script rate {cs_rate:.2%} outside ~10-15% India baseline (~5.0% overall)")
                        tier2_warnings += 1
                    if not (0.02 <= addr_missing <= 0.05):
                        print(f"  [WARN] India S3 missing address rate {addr_missing:.2%} outside ~3.3% tolerance")
                        tier2_warnings += 1

    # Transliteration coverage check across all partitions
    for fn, df in partition_dfs.items():
        cs_mask = df["is_cross_script"] == True
        if cs_mask.sum() > 0:
            cs_df = df[cs_mask]
            # When transliterated, name_for_bm25 should be in ASCII / different from name_for_faiss for Indic
            indic_mask = cs_df["script"].isin(["Devanagari", "Tamil", "Telugu", "Kannada", "Bengali", "Gujarati", "Gurmukhi", "Malayalam", "Oriya"])
            indic_df = cs_df[indic_mask]
            changed_count = (indic_df["name_for_bm25"] != indic_df["name_for_faiss"]).sum()
            total_indic = len(indic_df)
            print(f"{fn}: Indic cross-script rows={total_indic:,}, transliterated={changed_count:,} ({changed_count/max(1, total_indic):.2%})")

    if tier2_warnings == 0:
        print("[PASS] Tier 2 Distributional Checks within expected tolerance!")
    else:
        print(f"[WARN] Tier 2 completed with {tier2_warnings} warnings.")

    # ----------------------------------------------------
    # TIER 3: Spot-Sample Export
    # ----------------------------------------------------
    if spot_sample_out and cross_script_sample_rows:
        sample_df = pd.concat(cross_script_sample_rows, ignore_index=True).sample(n=min(50, sum(len(x) for x in cross_script_sample_rows)), random_state=42)
        sample_df.to_csv(spot_sample_out, sep="\t", index=False)
        print(f"\n[INFO] Tier 3 Spot Sample exported to: {spot_sample_out}")

    return tier1_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate Phase 0 parquet outputs")
    parser.add_argument("--processed-dir", default="/home/mivikev/Desktop/AMC/ML-Classification/data/processed")
    parser.add_argument("--dataset-dir", default="/home/mivikev/Desktop/AMC/Resource/dataset")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--spot-sample", default=None, help="Path to export Tier 3 spot sample TSV")
    args = parser.parse_args()

    spot_path = args.spot_sample
    if not spot_path:
        spot_path = f"/home/mivikev/Desktop/AMC/ML-Classification/metrics/phase0/spot_check_{args.split}.tsv"

    success = validate_split(
        processed_dir=args.processed_dir,
        raw_dataset_dir=args.dataset_dir,
        split=args.split,
        spot_sample_out=spot_path,
    )

    sys.exit(0 if success else 1)
