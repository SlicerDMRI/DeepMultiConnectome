#!/usr/bin/env python3
"""Compute network metric similarity between predicted and true connectomes."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from analysis.utils.network_metrics import compute_network_metrics


class NetworkMetricSimilarityAnalysis:
    def __init__(self, subject_list_file: str, max_subjects: int = None, min_length: float = None,
                 base_path: str = None, out_path: str = None,
                 atlases: List[str] = None, connectome_types: List[str] = None,
                 compute_advanced: bool = True, compute_community: bool = False):
        """
        Initialize network metric similarity analysis.
        
        Args:
            subject_list_file: Path to subject list file
            max_subjects: Max subjects to process
            min_length: Minimum streamline length filter
            base_path: Base data path (default: HCP_DATA_PATH env var or ./data)
            out_path: Output path (default: HCP_OUT_PATH env var or ./output)
            atlases: List of atlases to process
            connectome_types: List of connectome types to process
            compute_advanced: Whether to compute advanced metrics
            compute_community: Whether to compute community detection
        """
        if base_path is None:
            base_path = os.environ.get('HCP_DATA_PATH', './data')
        if out_path is None:
            out_path = os.environ.get('HCP_OUT_PATH', './output')
            
        self.subject_list_file = Path(subject_list_file)
        self.max_subjects = max_subjects
        self.min_length = min_length
        self.compute_advanced = compute_advanced
        self.compute_community = compute_community

        self.base_path = Path(base_path)
        self.results_dir = Path(out_path) / "network_metric_similarity"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()
        self.subjects = self._load_subject_list()

        self.atlases = atlases if atlases else ["aparc+aseg", "aparc.a2009s+aseg"]
        self.connectome_types = connectome_types if connectome_types else ["nos", "fa", "sift2"]

        self.log(f"Subjects: {len(self.subjects)}")
        self.log(f"Atlases: {self.atlases}")
        self.log(f"Connectome types: {self.connectome_types}")

    def _setup_logger(self) -> logging.Logger:
        log_file = self.results_dir / f"network_metrics_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logger = logging.getLogger("NetworkMetricSimilarity")
        logger.setLevel(logging.INFO)
        logger.handlers = []

        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

        return logger

    def log(self, message: str):
        self.logger.info(message)

    def _load_subject_list(self) -> List[str]:
        if not self.subject_list_file.exists():
            raise FileNotFoundError(f"Subject list file not found: {self.subject_list_file}")
        with open(self.subject_list_file, 'r') as f:
            subjects = [line.strip() for line in f if line.strip()]
        if self.max_subjects is not None:
            subjects = subjects[:self.max_subjects]
            self.log(f"Limited to first {self.max_subjects} subjects")
        return subjects

    def _analysis_dir(self, subject_id: str, atlas: str) -> Path:
        if self.min_length:
            return self.base_path / "HCP_MRtrix" / subject_id / "analysis" / f"{atlas}_minlen{int(self.min_length)}"
        return self.base_path / "HCP_MRtrix" / subject_id / "analysis" / atlas

    def _load_connectome_pair(self, subject_id: str, atlas: str, ctype: str):
        analysis_dir = self._analysis_dir(subject_id, atlas)
        true_file = analysis_dir / f"connectome_true_{ctype}_{atlas}.csv"
        pred_file = analysis_dir / f"connectome_pred_{ctype}_{atlas}.csv"
        if not (true_file.exists() and pred_file.exists()):
            return None, None
        try:
            t_mat = pd.read_csv(true_file, header=None).values
            p_mat = pd.read_csv(pred_file, header=None).values
            t_mat = np.nan_to_num(t_mat, nan=0.0)
            p_mat = np.nan_to_num(p_mat, nan=0.0)
            if t_mat.shape != p_mat.shape:
                return None, None
            if np.sum(t_mat) == 0 or np.sum(p_mat) == 0:
                return None, None
            return t_mat, p_mat
        except Exception:
            return None, None

    def run(self):
        rows = []
        for subject_id in self.subjects:
            for atlas in self.atlases:
                for ctype in self.connectome_types:
                    t_mat, p_mat = self._load_connectome_pair(subject_id, atlas, ctype)
                    if t_mat is None:
                        continue

                    true_metrics = compute_network_metrics(
                        t_mat,
                        compute_advanced=self.compute_advanced,
                        compute_community=self.compute_community
                    )
                    pred_metrics = compute_network_metrics(
                        p_mat,
                        compute_advanced=self.compute_advanced,
                        compute_community=self.compute_community
                    )

                    for metric_name, true_val in true_metrics.items():
                        pred_val = pred_metrics.get(metric_name, np.nan)
                        abs_diff = np.abs(pred_val - true_val)
                        rel_diff = abs_diff / (np.abs(true_val) + 1e-10)
                        sim_score = 1.0 - rel_diff
                        rows.append({
                            "subject_id": subject_id,
                            "atlas": atlas,
                            "connectome_type": ctype,
                            "metric": metric_name,
                            "true_value": true_val,
                            "pred_value": pred_val,
                            "abs_diff": abs_diff,
                            "rel_diff": rel_diff,
                            "similarity": sim_score
                        })

        metrics_df = pd.DataFrame(rows)
        if metrics_df.empty:
            self.log("No valid connectomes found.")
            return

        per_subject_path = self.results_dir / "network_metrics_per_subject.csv"
        metrics_df.to_csv(per_subject_path, index=False)
        self.log(f"Saved per-subject metrics: {per_subject_path}")

        summary_rows = []
        grouped = metrics_df.groupby(["atlas", "connectome_type", "metric"])
        for (atlas, ctype, metric), group in grouped:
            true_vals = group["true_value"].to_numpy()
            pred_vals = group["pred_value"].to_numpy()
            if len(true_vals) > 1 and np.std(true_vals) > 0 and np.std(pred_vals) > 0:
                corr = np.corrcoef(true_vals, pred_vals)[0, 1]
            else:
                corr = np.nan
            summary_rows.append({
                "atlas": atlas,
                "connectome_type": ctype,
                "metric": metric,
                "mean_abs_diff": float(group["abs_diff"].mean()),
                "mean_rel_diff": float(group["rel_diff"].mean()),
                "mean_similarity": float(group["similarity"].mean()),
                "corr_true_pred": corr
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = self.results_dir / "network_metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        self.log(f"Saved summary metrics: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network metric similarity analysis")
    parser.add_argument("--subject-list", required=True, help="Path to subject list file")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--min-length-mm", type=float, default=None)
    parser.add_argument("--atlases", nargs='+', default=None)
    parser.add_argument("--connectome-types", nargs='+', default=None)
    parser.add_argument("--no-advanced", action="store_true")
    parser.add_argument("--community", action="store_true")
    args = parser.parse_args()

    analysis = NetworkMetricSimilarityAnalysis(
        subject_list_file=args.subject_list,
        max_subjects=args.max_subjects,
        min_length=args.min_length_mm,
        atlases=args.atlases,
        connectome_types=args.connectome_types,
        compute_advanced=not args.no_advanced,
        compute_community=args.community
    )
    analysis.run()
