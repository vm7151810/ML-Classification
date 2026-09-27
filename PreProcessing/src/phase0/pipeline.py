"""Phase 0 Pipeline: Ingestion, Preprocessing, and Parquet Partitioning.

Streams raw source TSV files in chunks, applies open-set script detection,
name cleaning, transliteration dispatch, and address normalization, then
dynamically partitions and writes Query_<country>.parquet and Target_<country>.parquet.
"""

import os
import shutil
import time
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.phase0.script_detector import detect_script
from src.phase0.text_cleaner import check_has_address, clean_name_base, clean_address
from src.phase0.transliteration import transliterate_name
from src.phase0.metrics_logger import Phase0MetricsLogger

# Authoritative Arrow Schema per phase0_schema_contract.md
PHASE0_ARROW_SCHEMA = pa.schema([
    pa.field("entity_id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("country", pa.string()),
    pa.field("name_original", pa.string()),
    pa.field("addr_original", pa.string()),
    pa.field("is_cross_script", pa.bool_()),
    pa.field("script", pa.string()),
    pa.field("has_address", pa.bool_()),
    pa.field("name_for_faiss", pa.string()),
    pa.field("name_for_bm25", pa.string()),
    pa.field("addr_for_bm25", pa.string()),
])

# Expected row counts from EDA for parsing integrity assertions
EXPECTED_ROW_COUNTS = {
    "train_source1.tsv": 2_206_821,
    "train_source2.tsv": 5_034_616,
    "train_source3.tsv": 5_285_603,
    "test_source1.tsv": 1_732_544,
    "test_source2.tsv": 4_887_273,
    "test_source3.tsv": 5_082_316,
}


def process_chunk(
    chunk_df: pd.DataFrame,
    expected_source_prefix: str,
) -> Tuple[Dict[str, pa.Table], Dict[str, list]]:
    """Process a chunk of raw TSV records into partitioned Arrow tables and chunk stats.

    Args:
        chunk_df: DataFrame chunk with columns [entity_id, business_name, business_address, country]
        expected_source_prefix: Expected prefix (e.g. 'S1', 'S2', 'S3')

    Returns:
        (country_tables, chunk_metrics)
    """
    n_rows = len(chunk_df)
    eids = chunk_df["entity_id"].astype(str).tolist()
    names_raw = chunk_df["business_name"].fillna("").astype(str).tolist()
    addrs_raw = chunk_df["business_address"].fillna("").astype(str).tolist()
    countries = chunk_df["country"].astype(str).tolist()

    # Step 2: Derive source and assert prefix integrity
    sources = []
    for eid in eids:
        prefix = eid.split("-")[0]
        if prefix != expected_source_prefix:
            raise ValueError(
                f"Source prefix mismatch for entity_id '{eid}': expected '{expected_source_prefix}', got '{prefix}'"
            )
        sources.append(prefix)

    # Step 3: Script detection
    scripts = []
    cross_scripts = []
    for name in names_raw:
        scr, cs = detect_script(name)
        scripts.append(scr)
        cross_scripts.append(cs)

    # Step 4: Address presence pre-cleaning
    has_addrs = [check_has_address(addr) for addr in addrs_raw]

    # Step 5 & 6: Clean name (shared base pass & name_for_faiss)
    names_faiss = [clean_name_base(name) for name in names_raw]

    # Step 7: Transliteration dispatch for name_for_bm25
    names_bm25 = []
    translit_handled = []
    translit_degenerate = []
    for name_f, scr, cs in zip(names_faiss, scripts, cross_scripts):
        name_b, handled, degen = transliterate_name(name_f, scr, cs)
        names_bm25.append(name_b)
        translit_handled.append(handled)
        translit_degenerate.append(degen)

    # Step 8: Clean address for addr_for_bm25
    addrs_bm25 = [clean_address(addr, ha) for addr, ha in zip(addrs_raw, has_addrs)]

    # Compute length arrays for metrics
    name_clean_lens = [len(n) for n in names_faiss]
    addr_clean_lens = [len(a) for a in addrs_bm25]

    # Group by country dynamically
    countries_arr = np.array(countries)
    unique_countries = np.unique(countries_arr)

    country_tables: Dict[str, pa.Table] = {}
    chunk_metrics: Dict[str, dict] = {}

    for c in unique_countries:
        idx = np.where(countries_arr == c)[0]

        sub_eids = [eids[i] for i in idx]
        sub_sources = [sources[i] for i in idx]
        sub_countries = [countries[i] for i in idx]
        sub_names_raw = [names_raw[i] for i in idx]
        sub_addrs_raw = [addrs_raw[i] for i in idx]
        sub_cross_scripts = [cross_scripts[i] for i in idx]
        sub_scripts = [scripts[i] for i in idx]
        sub_has_addrs = [has_addrs[i] for i in idx]
        sub_names_faiss = [names_faiss[i] for i in idx]
        sub_names_bm25 = [names_bm25[i] for i in idx]
        sub_addrs_bm25 = [addrs_bm25[i] for i in idx]

        table = pa.Table.from_arrays(
            [
                pa.array(sub_eids, pa.string()),
                pa.array(sub_sources, pa.string()),
                pa.array(sub_countries, pa.string()),
                pa.array(sub_names_raw, pa.string()),
                pa.array(sub_addrs_raw, pa.string()),
                pa.array(sub_cross_scripts, pa.bool_()),
                pa.array(sub_scripts, pa.string()),
                pa.array(sub_has_addrs, pa.bool_()),
                pa.array(sub_names_faiss, pa.string()),
                pa.array(sub_names_bm25, pa.string()),
                pa.array(sub_addrs_bm25, pa.string()),
            ],
            schema=PHASE0_ARROW_SCHEMA,
        )
        country_tables[c] = table

        chunk_metrics[c] = {
            "sources": sub_sources,
            "scripts": sub_scripts,
            "cross_scripts": sub_cross_scripts,
            "has_addrs": sub_has_addrs,
            "translit_handled": [translit_handled[i] for i in idx],
            "translit_degenerate": [translit_degenerate[i] for i in idx],
            "name_clean_lens": [name_clean_lens[i] for i in idx],
            "addr_clean_lens": [addr_clean_lens[i] for i in idx],
        }

    return country_tables, chunk_metrics


def run_phase0_split(
    dataset_dir: str,
    output_dir: str,
    split: str,
    logger: Phase0MetricsLogger,
    chunksize: int = 250_000,
):
    """Run Phase 0 for a single split ('train' or 'test')."""
    split_dir = os.path.join(dataset_dir, split)
    split_out_dir = os.path.join(output_dir, split)
    os.makedirs(split_out_dir, exist_ok=True)

    # Remove existing parquet files to ensure a clean run
    for old_file in os.listdir(split_out_dir):
        if old_file.endswith(".parquet"):
            os.remove(os.path.join(split_out_dir, old_file))

    # Active Parquet writers: (role, country) -> ParquetWriter
    writers: Dict[Tuple[str, str], pq.ParquetWriter] = {}

    file_configs = [
        (f"{split}_source1.tsv", "S1", "Query"),
        (f"{split}_source2.tsv", "S2", "Target"),
        (f"{split}_source3.tsv", "S3", "Target"),
    ]

    try:
        for filename, prefix, role in file_configs:
            filepath = os.path.join(split_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Input file not found: {filepath}")

            expected_count = EXPECTED_ROW_COUNTS.get(filename)
            total_read_for_file = 0

            print(f"[{split.upper()}] Processing {filename} (role={role}, prefix={prefix})...")
            t_file_start = time.time()

            reader = pd.read_csv(
                filepath,
                sep="\t",
                chunksize=chunksize,
                dtype=str,
                keep_default_na=False,  # Treat NA as literal strings
            )

            for chunk_idx, chunk_df in enumerate(reader):
                chunk_len = len(chunk_df)
                total_read_for_file += chunk_len

                country_tables, chunk_metrics = process_chunk(chunk_df, prefix)

                for country, table in country_tables.items():
                    writer_key = (role, country)
                    if writer_key not in writers:
                        out_path = os.path.join(split_out_dir, f"{role}_{country}.parquet")
                        writers[writer_key] = pq.ParquetWriter(
                            out_path,
                            schema=PHASE0_ARROW_SCHEMA,
                            compression="snappy",
                        )

                    writers[writer_key].write_table(table)

                    # Record metrics
                    m = chunk_metrics[country]
                    logger.record_chunk(
                        source_name=filename,
                        country=country,
                        file_role=role,
                        sources=m["sources"],
                        scripts=m["scripts"],
                        cross_scripts=m["cross_scripts"],
                        has_addrs=m["has_addrs"],
                        translit_handled=m["translit_handled"],
                        translit_degenerate=m["translit_degenerate"],
                        name_clean_lens=m["name_clean_lens"],
                        addr_clean_lens=m["addr_clean_lens"],
                    )

            # Step 1: Parse integrity assertion
            if expected_count is not None and total_read_for_file != expected_count:
                raise ValueError(
                    f"Integrity check failed for {filename}: read {total_read_for_file} rows, expected {expected_count}"
                )

            elapsed = time.time() - t_file_start
            logger.timings[f"{split}_{filename}"] = round(elapsed, 2)
            print(f"[{split.upper()}] Finished {filename}: {total_read_for_file:,} rows in {elapsed:.1f}s")

    finally:
        # Ensure all writers are closed properly
        for writer in writers.values():
            writer.close()


def run_pipeline(
    dataset_dir: str = "/home/mivikev/Desktop/AMC/Resource/dataset",
    output_dir: str = "/home/mivikev/Desktop/AMC/ML-Classification/data/processed",
    metrics_dir: str = "/home/mivikev/Desktop/AMC/ML-Classification/metrics/phase0",
    splits: Optional[list] = None,
    chunksize: int = 250_000,
) -> str:
    """Run the complete Phase 0 pipeline for specified splits (default: ['train', 'test'])."""
    if splits is None:
        splits = ["train", "test"]

    logger = Phase0MetricsLogger(metrics_dir=metrics_dir, pipeline_version="0.1.0")
    t_global_start = time.time()

    for split in splits:
        t_split_start = time.time()
        run_phase0_split(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            split=split,
            logger=logger,
            chunksize=chunksize,
        )
        logger.timings[f"total_{split}"] = round(time.time() - t_split_start, 2)

    logger.timings["total_pipeline"] = round(time.time() - t_global_start, 2)
    saved_metrics_file = logger.save()
    print(f"Phase 0 complete. Metrics written to: {saved_metrics_file}")
    return saved_metrics_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 0 Preprocessing & Country Partitioning")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to process: train, test")
    parser.add_argument("--chunksize", type=int, default=250_000, help="Chunk size for streaming processing")
    args = parser.parse_args()

    run_pipeline(splits=args.splits, chunksize=args.chunksize)
