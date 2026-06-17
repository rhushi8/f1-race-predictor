"""
One-command evaluator for strict walk-forward calibration across multiple seasons.

Usage:
    python src/tuning/run_full_evaluation.py
    python src/tuning/run_full_evaluation.py --years 2021 2022 2023 2024
    python src/tuning/run_full_evaluation.py --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import PROC_DIR
from src.tuning.walk_forward_backtest import run_backtest
from src.tuning.calibration_report import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict walk-forward calibration sweeps and a global report"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(PROC_DIR / "historical_results.csv"),
        help="Path to historical CSV",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024],
        help="Seasons to evaluate (default: 2020..2024)",
    )
    parser.add_argument(
        "--min-train-races",
        type=int,
        default=5,
        help="Minimum number of races required in training before evaluating a round",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute backtests even if sweep CSV already exists",
    )
    return parser.parse_args()


def _has_existing_sweep(year: int) -> bool:
    sweep_path = PROC_DIR / f"walk_forward_{year}_calibration_sweep.csv"
    return sweep_path.exists()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"ERROR: historical CSV not found: {csv_path}")
        return 1

    years = sorted(set(args.years))

    print("\n" + "=" * 80)
    print("STRICT WALK-FORWARD ORCHESTRATOR".center(80))
    print("=" * 80)
    print(f"CSV:   {csv_path}")
    print(f"Years: {years}")
    print(f"Force: {args.force}")

    for year in years:
        if _has_existing_sweep(year) and not args.force:
            print(f"\n[{year}] Reusing existing sweep (use --force to recompute)")
            continue

        print(f"\n[{year}] Running strict walk-forward calibration sweep...")
        rc = run_backtest(
            csv_path=csv_path,
            year=year,
            optimize_calibration=True,
            lock_best=False,
            min_train_races=args.min_train_races,
        )
        if rc != 0:
            print(f"[{year}] Backtest failed with exit code {rc}")
            return rc

    print("\n" + "=" * 80)
    print("AGGREGATED REPORT".center(80))
    print("=" * 80)
    generate_report(years)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
