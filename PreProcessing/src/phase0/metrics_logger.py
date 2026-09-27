"""Metrics tracking and structured logging for Phase 0.

Produces structured JSON records under metrics/phase0/<run_id>.json and appends
to metrics/phase0/history.jsonl per phase0_metrics_and_logging.md specifications.
"""

from collections import defaultdict
import datetime
import json
import os
from typing import Dict, Any, List
import numpy as np


class Phase0MetricsLogger:
    """Collects and writes structured run metrics for Phase 0."""

    def __init__(self, metrics_dir: str, pipeline_version: str = "0.1.0"):
        self.metrics_dir = metrics_dir
        self.pipeline_version = pipeline_version
        self.run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        os.makedirs(metrics_dir, exist_ok=True)

        # Stage timings
        self.timings: Dict[str, float] = {}

        # Per source file counts
        self.source_input_rows: Dict[str, int] = defaultdict(int)
        self.source_output_rows: Dict[str, int] = defaultdict(int)

        # Per country & file type counts (Query / Target)
        self.partition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Cross-script counts
        self.cross_script_by_country: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "cross_script": 0})
        self.cross_script_by_source: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "cross_script": 0})

        # Script distribution by country
        self.script_distribution_by_country: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Address presence counts
        self.address_by_country: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "has_address": 0})
        self.address_by_source: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "has_address": 0})

        # Transliteration tracking
        # script -> {attempted, handled, unhandled, degenerate}
        self.transliteration_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"attempted": 0, "handled": 0, "unhandled": 0, "degenerate": 0}
        )

        # Length tracking post-cleaning for percentiles
        self._name_lengths_by_country: Dict[str, List[int]] = defaultdict(list)
        self._addr_lengths_by_country: Dict[str, List[int]] = defaultdict(list)

    def record_chunk(
        self,
        source_name: str,
        country: str,
        file_role: str,  # "Query" or "Target"
        sources: List[str],
        scripts: List[str],
        cross_scripts: List[bool],
        has_addrs: List[bool],
        translit_handled: List[bool],
        translit_degenerate: List[bool],
        name_clean_lens: List[int],
        addr_clean_lens: List[int],
    ):
        """Record chunk-level metrics in aggregate counters."""
        n_rows = len(sources)
        self.source_input_rows[source_name] += n_rows
        self.source_output_rows[source_name] += n_rows
        self.partition_counts[country][file_role] += n_rows

        c_cross = self.cross_script_by_country[country]
        c_addr = self.address_by_country[country]
        c_cross["total"] += n_rows
        c_addr["total"] += n_rows

        scripts_by_c = self.script_distribution_by_country[country]

        for s, scr, cs, ha, th, td in zip(
            sources,
            scripts,
            cross_scripts,
            has_addrs,
            translit_handled,
            translit_degenerate,
        ):
            s_cross = self.cross_script_by_source[s]
            s_addr = self.address_by_source[s]
            s_cross["total"] += 1
            s_addr["total"] += 1

            scripts_by_c[scr] += 1

            if cs:
                c_cross["cross_script"] += 1
                s_cross["cross_script"] += 1

                t_stat = self.transliteration_stats[scr]
                t_stat["attempted"] += 1
                if th:
                    t_stat["handled"] += 1
                else:
                    t_stat["unhandled"] += 1
                if td:
                    t_stat["degenerate"] += 1

            if ha:
                c_addr["has_address"] += 1
                s_addr["has_address"] += 1

        # Keep a reservoir/sample of lengths for percentile estimation to save memory
        # Sample up to 10,000 lengths per country
        if len(self._name_lengths_by_country[country]) < 10000:
            sample_step = max(1, n_rows // 500)
            self._name_lengths_by_country[country].extend(name_clean_lens[::sample_step])
            self._addr_lengths_by_country[country].extend(addr_clean_lens[::sample_step])

    def compute_summary(self) -> Dict[str, Any]:
        """Compile aggregated metrics into final serializable dictionary."""
        # Rates computation
        cross_script_rates = {
            "by_country": {
                c: (v["cross_script"] / v["total"] if v["total"] > 0 else 0.0)
                for c, v in self.cross_script_by_country.items()
            },
            "by_source": {
                s: (v["cross_script"] / v["total"] if v["total"] > 0 else 0.0)
                for s, v in self.cross_script_by_source.items()
            },
        }

        has_address_rates = {
            "by_country": {
                c: (v["has_address"] / v["total"] if v["total"] > 0 else 0.0)
                for c, v in self.address_by_country.items()
            },
            "by_source": {
                s: (v["has_address"] / v["total"] if v["total"] > 0 else 0.0)
                for s, v in self.address_by_source.items()
            },
        }

        # Percentiles
        length_percentiles = {}
        for country in self._name_lengths_by_country:
            names = np.array(self._name_lengths_by_country[country])
            addrs = np.array(self._addr_lengths_by_country[country])
            length_percentiles[country] = {
                "name": {
                    "P25": float(np.percentile(names, 25)) if len(names) else 0.0,
                    "P50": float(np.percentile(names, 50)) if len(names) else 0.0,
                    "P75": float(np.percentile(names, 75)) if len(names) else 0.0,
                },
                "address": {
                    "P25": float(np.percentile(addrs, 25)) if len(addrs) else 0.0,
                    "P50": float(np.percentile(addrs, 50)) if len(addrs) else 0.0,
                    "P75": float(np.percentile(addrs, 75)) if len(addrs) else 0.0,
                },
            }

        return {
            "run_id": self.run_id,
            "pipeline_version": self.pipeline_version,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "timings_seconds": self.timings,
            "source_files": {
                s: {
                    "input_rows": self.source_input_rows[s],
                    "output_rows": self.source_output_rows[s],
                    "match": self.source_input_rows[s] == self.source_output_rows[s],
                }
                for s in self.source_input_rows
            },
            "partition_counts": {c: dict(v) for c, v in self.partition_counts.items()},
            "cross_script_rate": cross_script_rates,
            "script_distribution": {
                c: dict(v) for c, v in self.script_distribution_by_country.items()
            },
            "has_address_rate": has_address_rates,
            "transliteration_coverage": {
                scr: dict(stats) for scr, stats in self.transliteration_stats.items()
            },
            "length_percentiles": length_percentiles,
        }

    def save(self) -> str:
        """Save the metrics JSON file and append to history.jsonl."""
        summary = self.compute_summary()
        run_file = os.path.join(self.metrics_dir, f"{self.run_id}.json")
        history_file = os.path.join(self.metrics_dir, "history.jsonl")

        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        return run_file
