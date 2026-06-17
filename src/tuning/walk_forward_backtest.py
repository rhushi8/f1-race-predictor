"""
Strict walk-forward evaluator for season-level out-of-sample validation.

Usage examples:
    python src/tuning/walk_forward_backtest.py --year 2024
    python src/tuning/walk_forward_backtest.py --year 2024 --optimize-calibration
    python src/tuning/walk_forward_backtest.py --year 2024 --optimize-calibration --lock-best
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import CALIBRATION_PARAMS, PROC_DIR
from src.features.engineer import build_feature_matrix
from src.models.ensemble import F1StackingEnsemble

log = logging.getLogger(__name__)


def _qualifying_rank_from_df(df: pd.DataFrame) -> np.ndarray:
    if "grid_position" in df.columns and not df["grid_position"].isna().all():
        return pd.to_numeric(df["grid_position"], errors="coerce").to_numpy(dtype=float)

    for col in ["q3_gap_to_pole", "q2_gap_to_pole", "q1_gap_to_pole"]:
        if col in df.columns and not df[col].isna().all():
            return df[col].rank(method="first", ascending=True).to_numpy(dtype=float)

    return np.arange(1, len(df) + 1, dtype=float)


def _blend_positions(model_pos: np.ndarray, test_df: pd.DataFrame, model_weight: float) -> np.ndarray:
    qual_rank = _qualifying_rank_from_df(test_df)
    model_pos = np.asarray(model_pos, dtype=float)

    # Guard against sparse qualifying/grid inputs and occasional model NaNs.
    if np.isnan(model_pos).any():
        finite_mask = np.isfinite(model_pos)
        fallback = float(np.nanmean(model_pos[finite_mask])) if finite_mask.any() else 10.0
        model_pos = np.where(np.isfinite(model_pos), model_pos, fallback)

    if np.isnan(qual_rank).any():
        qual_rank = np.where(np.isfinite(qual_rank), qual_rank, model_pos)

    blended = model_weight * model_pos + (1.0 - model_weight) * qual_rank
    return np.clip(blended, 1.0, 20.0)


def _evaluate_for_weight(
    season_df: pd.DataFrame,
    rounds: list[int],
    model_weight: float,
    min_train_races: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    # Predict round n with training data from rounds < n only.
    for rnd in rounds[1:]:
        train_df = season_df[season_df["round"] < rnd].copy()
        test_df = season_df[season_df["round"] == rnd].copy()

        if train_df[["year", "gp"]].drop_duplicates().shape[0] < min_train_races:
            continue
        if test_df.empty:
            continue

        train_result = build_feature_matrix(train_df, historical_results=train_df, fit=True)
        X_train, artifacts = train_result

        y_train = pd.to_numeric(train_df["finish_position"], errors="coerce").astype(float)
        y_dnf_train = pd.to_numeric(train_df.get("dnf", 0), errors="coerce").fillna(0).astype(int)

        model = F1StackingEnsemble(n_splits=5)
        model.fit(X_train, y_train, y_dnf=y_dnf_train, y_sc=None)

        test_result = build_feature_matrix(
            test_df,
            historical_results=train_df,
            fit=False,
            artifacts=artifacts,
            feature_cols=getattr(model, "feature_cols", None),
        )
        X_test = test_result.X
        preds = model.predict(X_test)

        model_pos = np.asarray(preds["position_pred"], dtype=float)
        pred_pos = _blend_positions(model_pos, test_df, model_weight=model_weight)
        actual_pos = pd.to_numeric(test_df["finish_position"], errors="coerce").to_numpy(dtype=float)

        mae = float(mean_absolute_error(actual_pos, pred_pos))
        rmse = float(np.sqrt(mean_squared_error(actual_pos, pred_pos)))

        tmp = test_df[["driver_code", "finish_position"]].copy()
        tmp["pred_pos"] = pred_pos
        pred_top10 = set(tmp.sort_values("pred_pos").head(10)["driver_code"])
        actual_top10 = set(tmp.sort_values("finish_position").head(10)["driver_code"])
        top10_hit = len(pred_top10 & actual_top10) / 10.0

        rows.append(
            {
                "round": int(rnd),
                "gp": str(test_df["gp"].iloc[0]),
                "mae": mae,
                "rmse": rmse,
                "top10_hit": top10_hit,
                "weight": float(model_weight),
            }
        )

    return pd.DataFrame(rows)


def _lock_weight_in_settings(weight: float, settings_path: Path) -> bool:
    text = settings_path.read_text(encoding="utf-8")
    pattern = r'("model_position_weight"\s*:\s*)([0-9]*\.?[0-9]+)'

    if re.search(pattern, text) is None:
        log.warning("Could not find model_position_weight in %s", settings_path)
        return False

    updated = re.sub(pattern, rf"\g<1>{weight:.2f}", text, count=1)
    if updated == text:
        return False

    settings_path.write_text(updated, encoding="utf-8")
    return True


def run_backtest(
    csv_path: Path,
    year: int,
    optimize_calibration: bool,
    lock_best: bool,
    min_train_races: int,
) -> int:
    df = pd.read_csv(csv_path)
    needed = {"year", "round", "gp", "driver_code", "finish_position"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    season_df = df[df["year"] == year].copy()
    season_df["round"] = pd.to_numeric(season_df["round"], errors="coerce")
    season_df["finish_position"] = pd.to_numeric(season_df["finish_position"], errors="coerce")
    season_df = season_df.dropna(subset=["round", "finish_position", "driver_code"])
    season_df["round"] = season_df["round"].astype(int)

    rounds = sorted(season_df["round"].unique().tolist())
    if len(rounds) < 3:
        raise RuntimeError(f"Need at least 3 rounds for year={year}; found {len(rounds)}")

    if optimize_calibration:
        weights = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
    else:
        weights = [float(CALIBRATION_PARAMS.get("model_position_weight", 1.0))]

    sweep_rows: list[dict[str, float | int]] = []
    best_metrics: tuple[float, float, float] | None = None
    best_weight = weights[0]
    best_per_race: pd.DataFrame | None = None

    for w in weights:
        per_race = _evaluate_for_weight(season_df, rounds, model_weight=w, min_train_races=min_train_races)
        if per_race.empty:
            continue

        mean_mae = float(per_race["mae"].mean())
        mean_rmse = float(per_race["rmse"].mean())
        mean_top10 = float(per_race["top10_hit"].mean())
        sweep_rows.append({"weight": w, "mae": mean_mae, "rmse": mean_rmse, "top10_hit": mean_top10, "rounds": len(per_race)})
        print(f"weight={w:>4.2f} | rounds={len(per_race):2d} | MAE={mean_mae:.3f} RMSE={mean_rmse:.3f} Top10={mean_top10:.3f}")

        if (best_metrics is None) or (mean_mae < best_metrics[0]):
            best_metrics = (mean_mae, mean_rmse, mean_top10)
            best_weight = w
            best_per_race = per_race

    if best_metrics is None or best_per_race is None:
        raise RuntimeError("No eligible rounds evaluated. Try lowering --min-train-races.")

    per_race_out = PROC_DIR / f"walk_forward_{year}_metrics.csv"
    best_per_race.to_csv(per_race_out, index=False)

    sweep_df = pd.DataFrame(sweep_rows).sort_values("weight").reset_index(drop=True)
    sweep_out = PROC_DIR / f"walk_forward_{year}_calibration_sweep.csv"
    sweep_df.to_csv(sweep_out, index=False)

    print("\n=== BEST (strict walk-forward) ===")
    print(f"Year: {year}")
    print(f"Best weight: {best_weight:.2f}")
    print(f"Mean MAE:  {best_metrics[0]:.3f}")
    print(f"Mean RMSE: {best_metrics[1]:.3f}")
    print(f"Mean Top10 hit: {best_metrics[2]:.3f}")
    print(f"Saved per-race: {per_race_out}")
    print(f"Saved sweep:    {sweep_out}")

    if lock_best:
        current_weight = float(CALIBRATION_PARAMS.get("model_position_weight", 1.0))
        current_row = sweep_df.loc[np.isclose(sweep_df["weight"], current_weight)]
        current_mae = float(current_row["mae"].iloc[0]) if not current_row.empty else float("inf")

        if best_metrics[0] + 1e-9 < current_mae:
            settings_path = Path(__file__).parent.parent.parent / "config" / "settings.py"
            changed = _lock_weight_in_settings(best_weight, settings_path)
            if changed:
                print(
                    f"Locked new default model_position_weight={best_weight:.2f} "
                    f"(strict MAE improved {current_mae:.3f} -> {best_metrics[0]:.3f})."
                )
            else:
                print("Best weight improved MAE, but config update could not be applied automatically.")
        else:
            print(
                f"Kept existing default model_position_weight={current_weight:.2f} "
                f"(best strict MAE {best_metrics[0]:.3f}, current strict MAE {current_mae:.3f})."
            )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict walk-forward backtest for F1 predictor")
    parser.add_argument("--csv", type=str, default=str(PROC_DIR / "historical_results.csv"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--min-train-races", type=int, default=5)
    parser.add_argument("--optimize-calibration", action="store_true")
    parser.add_argument("--lock-best", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_backtest(
            csv_path=Path(args.csv),
            year=args.year,
            optimize_calibration=args.optimize_calibration,
            lock_best=args.lock_best,
            min_train_races=args.min_train_races,
        )
    )