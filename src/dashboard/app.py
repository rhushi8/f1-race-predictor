"""
src/dashboard/app.py

Portfolio-ready Formula 1 Race Prediction Dashboard.

Usage:
    python src/dashboard/app.py
    python src/dashboard/app.py --predictions data/processed/2025_Australian_Grand_Prix/predictions.csv
    python src/dashboard/app.py --port 8051 --debug
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Input, Output, State, callback, dcc, html, dash_table
import dash_bootstrap_components as dbc

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROC_DIR


# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
TEAM_COLORS = {
    "Red Bull": "#1E5BC6",
    "Ferrari": "#E10600",
    "Mercedes": "#00D2BE",
    "McLaren": "#FF8000",
    "Aston Martin": "#2D826D",
    "Alpine": "#2293D1",
    "Williams": "#005AFF",
    "RB": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas": "#B6BABD",
    "Sauber": "#52E252",
    "Audi": "#6C757D",
    "Cadillac": "#1F4EA8",
}

THEME = {
    "bg": "#0F1218",
    "panel": "#171C24",
    "panel_alt": "#1D2430",
    "border": "#2C3443",
    "text": "#F5F7FA",
    "muted": "#9CA8BC",
    "accent": "#E10600",
    "accent_soft": "#FF6B6B",
    "good": "#00C389",
}

DEFAULT_PREDICTION_SEASON = 2026
GRID_SIZE_DEFAULT = 22

DRIVER_TEAM_FALLBACK = {
    "VER": "Red Bull",
    "PER": "Red Bull",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "DOO": "Alpine",
    "ALB": "Williams",
    "SAI": "Williams",
    "OCO": "Haas",
    "BEA": "Haas",
    "TSU": "RB",
    "LAW": "RB",
    "HUL": "Kick Sauber",
    "BOR": "Kick Sauber",
    "HAD": "RB",
    "VES": "Red Bull",
    "DRU": "Aston Martin",
    "IWA": "RB",
    "HIR": "Alpine",
    "BEG": "Haas",
    "BRO": "Kick Sauber",
    "COL": "Cadillac",
    "ZHO": "Audi",
}

DRIVER_NAME_FALLBACK = {
    "VER": "Max Verstappen",
    "PER": "Sergio Perez",
    "LEC": "Charles Leclerc",
    "HAM": "Lewis Hamilton",
    "RUS": "George Russell",
    "ANT": "Kimi Antonelli",
    "NOR": "Lando Norris",
    "PIA": "Oscar Piastri",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",
    "DOO": "Jack Doohan",
    "ALB": "Alex Albon",
    "SAI": "Carlos Sainz",
    "OCO": "Esteban Ocon",
    "BEA": "Oliver Bearman",
    "TSU": "Yuki Tsunoda",
    "LAW": "Liam Lawson",
    "HUL": "Nico Hulkenberg",
    "BOR": "Gabriel Bortoleto",
    "HAD": "Isack Hadjar",
    "VES": "Frederik Vesti",
    "DRU": "Felipe Drugovich",
    "IWA": "Ayumu Iwasa",
    "HIR": "Ryo Hirakawa",
    "BEG": "Paul Aron",
    "BRO": "Zak Brown",
    "COL": "Franco Colapinto",
    "ZHO": "Zhou Guanyu",
}

# Canonicalize reserve/alias codes to primary season roster slots for consistency.
DRIVER_CODE_ALIAS = {
    # Keep this minimal: only collapse clearly invalid non-driver codes.
    "BRO": "BOR",
}


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lookup:
            return lookup[key]
    return None


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _infer_race_meta_from_path(path: Path) -> tuple[Optional[int], Optional[str]]:
    parent_name = path.parent.name
    m = re.match(r"^(\d{4})_(.+)$", parent_name)
    if not m:
        return None, None
    year = int(m.group(1))
    race = m.group(2).replace("_", " ")
    return year, race


def _normalize_constructor_name(name: str) -> str:
    n = str(name).strip()
    if not n:
        return "Unknown"
    low = n.lower()
    if "red bull" in low:
        return "Red Bull"
    if "ferrari" in low:
        return "Ferrari"
    if "mercedes" in low:
        return "Mercedes"
    if "mclaren" in low:
        return "McLaren"
    if "aston" in low:
        return "Aston Martin"
    if "alpine" in low:
        return "Alpine"
    if "williams" in low:
        return "Williams"
    if "rb" in low or "racing bulls" in low or "toro rosso" in low:
        return "RB"
    if "haas" in low:
        return "Haas"
    if "audi" in low or "sauber" in low or "stake" in low or "alfa" in low:
        return "Audi"
    if "cadillac" in low:
        return "Cadillac"
    return n


def _build_latest_driver_team_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    hist_candidates = [
        PROC_DIR / "historical_results_2025_extended.csv",
        PROC_DIR / "historical_results.csv",
    ]
    hist = _first_existing(hist_candidates)
    if hist is None:
        return lookup

    try:
        h = pd.read_csv(hist)
    except Exception:
        return lookup

    dcol = _find_col(h, ["driver_code", "driver"])
    tcol = _find_col(h, ["team_name", "team", "constructor"])
    ycol = _find_col(h, ["year", "season"])
    rcol = _find_col(h, ["round", "race_number_in_season", "race_round"])

    if dcol is None or tcol is None:
        return lookup

    hh = h[[c for c in [dcol, tcol, ycol, rcol] if c is not None]].copy()
    hh[dcol] = hh[dcol].astype(str).str.upper()
    hh[tcol] = hh[tcol].astype(str)

    sort_cols = []
    if ycol is not None:
        hh[ycol] = pd.to_numeric(hh[ycol], errors="coerce").fillna(0)
        sort_cols.append(ycol)
    if rcol is not None:
        hh[rcol] = pd.to_numeric(hh[rcol], errors="coerce").fillna(0)
        sort_cols.append(rcol)

    if sort_cols:
        hh = hh.sort_values(sort_cols)

    latest = hh.groupby(dcol, as_index=False).tail(1)
    for _, row in latest.iterrows():
        team = row[tcol]
        code = row[dcol]
        if pd.isna(team) or pd.isna(code):
            continue
        code = str(code).strip().upper()
        team = str(team).strip()
        if not code or not team or team.lower() in {"nan", "none"}:
            continue
        if "Red Bull" in team:
            team = "Red Bull"
        elif "RB" in team or "Racing Bulls" in team or "Toro Rosso" in team:
            team = "RB"
        elif "Sauber" in team or "Stake" in team or "Alfa" in team:
            team = "Kick Sauber"
        elif "Haas" in team:
            team = "Haas"
        elif "Mercedes" in team:
            team = "Mercedes"
        elif "Ferrari" in team:
            team = "Ferrari"
        elif "McLaren" in team:
            team = "McLaren"
        elif "Aston" in team:
            team = "Aston Martin"
        elif "Alpine" in team or "Renault" in team:
            team = "Alpine"
        elif "Williams" in team:
            team = "Williams"
        lookup[code] = str(team)

    return lookup


def _build_latest_driver_name_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    hist_candidates = [
        PROC_DIR / "historical_results_2025_extended.csv",
        PROC_DIR / "historical_results.csv",
    ]
    hist = _first_existing(hist_candidates)
    if hist is None:
        return lookup

    try:
        h = pd.read_csv(hist)
    except Exception:
        return lookup

    dcol = _find_col(h, ["driver_code", "driver"])
    ncol = _find_col(h, ["driver_name", "name"])
    ycol = _find_col(h, ["year", "season"])
    rcol = _find_col(h, ["round", "race_number_in_season", "race_round"])

    if dcol is None or ncol is None:
        return lookup

    hh = h[[c for c in [dcol, ncol, ycol, rcol] if c is not None]].copy()
    hh[dcol] = hh[dcol].astype(str).str.strip().str.upper()
    hh[ncol] = hh[ncol].astype(str).str.strip()

    sort_cols = []
    if ycol is not None:
        hh[ycol] = pd.to_numeric(hh[ycol], errors="coerce").fillna(0)
        sort_cols.append(ycol)
    if rcol is not None:
        hh[rcol] = pd.to_numeric(hh[rcol], errors="coerce").fillna(0)
        sort_cols.append(rcol)
    if sort_cols:
        hh = hh.sort_values(sort_cols)

    latest = hh.groupby(dcol, as_index=False).tail(1)
    for _, row in latest.iterrows():
        code = str(row[dcol]).strip().upper()
        name = str(row[ncol]).strip()
        if not code or not name or name.lower() in {"nan", "none"}:
            continue
        lookup[code] = name

    return lookup


def _load_predictions_from_processed_dirs(prediction_season: int) -> Optional[pd.DataFrame]:
    pred_files = sorted(PROC_DIR.glob("*_Grand_Prix/predictions.csv"))
    if not pred_files:
        return None

    team_lookup = _build_latest_driver_team_lookup()
    name_lookup = _build_latest_driver_name_lookup()
    rows: list[pd.DataFrame] = []

    for i, f in enumerate(pred_files, start=1):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue

        if d.empty:
            continue

        d = d.copy()
        if "driver_code" in d.columns and "driver" not in d.columns:
            d["driver"] = d["driver_code"].astype(str).str.upper()
        elif "driver" in d.columns:
            d["driver"] = d["driver"].astype(str).str.upper()
        else:
            continue

        d["driver"] = d["driver"].replace(DRIVER_CODE_ALIAS)

        src_year, race_name = _infer_race_meta_from_path(f)
        d["season"] = prediction_season if prediction_season is not None else (src_year or DEFAULT_PREDICTION_SEASON)
        d["race"] = race_name if race_name else f"Round {i}"
        d["race_round"] = i

        if "constructor" not in d.columns and "team" in d.columns:
            d["constructor"] = d["team"].astype(str)
        if "constructor" not in d.columns:
            d["constructor"] = d["driver"].map(team_lookup)
            d["constructor"] = d["constructor"].fillna(d["driver"].map(DRIVER_TEAM_FALLBACK)).fillna("Unknown")
            d["constructor"] = d["constructor"].astype(str).str.strip()
            d.loc[d["constructor"].str.lower().isin(["nan", "none", ""]), "constructor"] = "Unknown"

        if "driver_name" not in d.columns:
            d["driver_name"] = d["driver"].map(name_lookup)
            d["driver_name"] = d["driver_name"].fillna(d["driver"].map(DRIVER_NAME_FALLBACK)).fillna(d["driver"])

        if "qualifying_position" not in d.columns:
            if "grid_position" in d.columns:
                d["qualifying_position"] = d["grid_position"]
            else:
                d["qualifying_position"] = d["expected_position"].round() if "expected_position" in d.columns else 10

        if "grid_position" not in d.columns:
            d["grid_position"] = d["qualifying_position"]

        if "finish_position" not in d.columns:
            d["finish_position"] = d["expected_position"].round() if "expected_position" in d.columns else 10

        if "points" not in d.columns:
            fp = pd.to_numeric(d["finish_position"], errors="coerce").fillna(99).astype(int)
            d["points"] = fp.map({1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}).fillna(0)

        if "pit_stops" not in d.columns:
            d["pit_stops"] = 2
        if "avg_lap_time_s" not in d.columns:
            d["avg_lap_time_s"] = 91.5
        if "team_performance" not in d.columns:
            d["team_performance"] = 75
        if "historical_form" not in d.columns:
            d["historical_form"] = 75
        if "win_probability" not in d.columns and "win_prob" in d.columns:
            d["win_probability"] = d["win_prob"]
        if "confidence" not in d.columns:
            d["confidence"] = 0.5

        rows.append(d)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)

    # Use official 2026 R1-R2 results to lock roster/team/name consistency.
    ref_path = PROC_DIR / "2026_r1_r2_results.csv"
    if prediction_season == 2026 and ref_path.exists():
        ref = pd.read_csv(ref_path)
        if not ref.empty:
            ref = ref.copy()
            ref["driver"] = ref["driver"].astype(str).str.upper().replace(DRIVER_CODE_ALIAS)
            ref["driver_name"] = ref["driver_name"].astype(str)
            ref["constructor"] = ref["constructor"].apply(_normalize_constructor_name)
            ref["race"] = ref["race"].astype(str)

            ref_latest = ref[["driver", "driver_name", "constructor"]].drop_duplicates().copy()
            ref_latest = ref_latest.groupby("driver", as_index=False).tail(1)
            ref_name_map = dict(zip(ref_latest["driver"], ref_latest["driver_name"]))
            ref_team_map = dict(zip(ref_latest["driver"], ref_latest["constructor"]))
            ref_driver_set = set(ref_latest["driver"].tolist())

            combined["driver"] = combined["driver"].astype(str).str.upper().replace(DRIVER_CODE_ALIAS)
            combined["constructor"] = combined["constructor"].apply(_normalize_constructor_name)
            combined["driver_name"] = combined.get("driver_name", combined["driver"])
            combined["driver_name"] = combined["driver_name"].where(
                combined["driver_name"].astype(str).str.lower().ne(combined["driver"].astype(str).str.lower()),
                combined["driver"].map(ref_name_map).fillna(combined["driver_name"]),
            )
            combined["constructor"] = combined["driver"].map(ref_team_map).fillna(combined["constructor"])
            combined = combined[combined["driver"].isin(ref_driver_set)].copy()

            # Overwrite known races with actual outcomes for calibration quality.
            ref_actual = ref[["race", "driver", "grid_position", "finish_position", "points"]].copy()
            ref_actual = ref_actual.rename(
                columns={
                    "grid_position": "actual_grid_position",
                    "finish_position": "actual_finish_position",
                    "points": "actual_points",
                }
            )
            combined = combined.merge(ref_actual, on=["race", "driver"], how="left")
            for c_actual, c_target in [
                ("actual_grid_position", "grid_position"),
                ("actual_finish_position", "finish_position"),
                ("actual_points", "points"),
            ]:
                combined[c_target] = pd.to_numeric(combined[c_actual], errors="coerce").fillna(
                    pd.to_numeric(combined[c_target], errors="coerce")
                )
            combined = combined.drop(columns=["actual_grid_position", "actual_finish_position", "actual_points"], errors="ignore")

            # Ensure each race has the full official grid where possible.
            added_rows: list[pd.DataFrame] = []
            key_cols = ["season", "race_round", "race"]
            for key, grp in combined.groupby(key_cols, as_index=False):
                present = set(grp["driver"].astype(str).str.upper().tolist())
                missing = sorted(ref_driver_set - present)
                if not missing:
                    continue
                season, race_round, race = key
                add_df = pd.DataFrame(
                    {
                        "season": season,
                        "race_round": race_round,
                        "race": race,
                        "driver": missing,
                        "driver_name": [ref_name_map.get(d, d) for d in missing],
                        "constructor": [ref_team_map.get(d, "Unknown") for d in missing],
                        "qualifying_position": GRID_SIZE_DEFAULT,
                        "grid_position": GRID_SIZE_DEFAULT,
                        "finish_position": GRID_SIZE_DEFAULT,
                        "points": 0.0,
                        "pit_stops": 2.0,
                        "avg_lap_time_s": 92.5,
                        "team_performance": 70.0,
                        "historical_form": 70.0,
                        "win_probability": 0.01,
                        "confidence": 0.45,
                    }
                )
                added_rows.append(add_df)

            if added_rows:
                combined = pd.concat([combined] + added_rows, ignore_index=True)

    combined = _limit_to_race_lineup(combined, max_drivers=GRID_SIZE_DEFAULT)
    combined = _calibrate_predictions_from_first_two_races(combined)
    return combined


def _limit_to_race_lineup(df: pd.DataFrame, max_drivers: int = GRID_SIZE_DEFAULT) -> pd.DataFrame:
    """Keep race lineups realistic by retaining top expected starters per race."""
    if df.empty:
        return df

    work = df.copy()
    if "win_probability" in work.columns:
        score = pd.to_numeric(work["win_probability"], errors="coerce").fillna(-1)
    else:
        score = -pd.to_numeric(work.get("expected_position", 99), errors="coerce").fillna(99)
    work["_lineup_score"] = score

    work = (
        work.sort_values(["season", "race_round", "driver", "_lineup_score"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["season", "race_round", "driver"], keep="first")
    )

    work = (
        work.sort_values(["season", "race_round", "_lineup_score"], ascending=[True, True, False])
        .groupby(["season", "race_round"], as_index=False)
        .head(max_drivers)
        .drop(columns=["_lineup_score"])
        .reset_index(drop=True)
    )

    # Keep a stable season roster (top grid size by participation/performance) to avoid
    # stand-in/test drivers inflating unique-driver counts across the year.
    roster_rank = (
        work.groupby("driver", as_index=False)
        .agg(
            race_count=("race", "nunique"),
            mean_win=("win_probability", "mean"),
            mean_finish=("finish_position", "mean"),
        )
        .sort_values(["race_count", "mean_win", "mean_finish"], ascending=[False, False, True])
    )
    keep_drivers = set(roster_rank.head(max_drivers)["driver"].tolist())
    work = work[work["driver"].isin(keep_drivers)].copy()

    return work


def _softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    z = x - np.nanmax(x)
    e = np.exp(z)
    denom = np.nansum(e)
    if denom <= 0:
        return np.ones_like(x) / len(x)
    return e / denom


def _calibrate_predictions_from_first_two_races(df: pd.DataFrame) -> pd.DataFrame:
    """Calibrate probabilities using the first two race results as early-season signal."""
    if df.empty or "race_round" not in df.columns:
        return df

    work = df.copy()
    work["win_probability"] = pd.to_numeric(work.get("win_probability", 0), errors="coerce").fillna(0)
    work["qualifying_position"] = pd.to_numeric(work.get("qualifying_position", 10), errors="coerce").fillna(10)
    work["finish_position"] = pd.to_numeric(work.get("finish_position", 10), errors="coerce").fillna(10)

    ref_rounds = sorted(work["race_round"].dropna().unique().tolist())[:2]
    ref = work[work["race_round"].isin(ref_rounds)].copy()
    if ref.empty:
        return work

    team_ref = (
        ref.groupby("constructor", as_index=False)
        .agg(
            team_ref_win=("win_probability", "mean"),
            team_ref_finish=("finish_position", "mean"),
            team_ref_points=("points", "mean"),
            team_ref_pace=("avg_lap_time_s", "mean"),
        )
    )
    driver_ref = (
        ref.groupby("driver", as_index=False)
        .agg(
            driver_ref_win=("win_probability", "mean"),
            driver_ref_finish=("finish_position", "mean"),
            driver_ref_points=("points", "mean"),
        )
    )

    work = work.merge(team_ref, on="constructor", how="left")
    work = work.merge(driver_ref, on="driver", how="left")

    for col, fallback in [
        ("team_ref_win", work["win_probability"].mean()),
        ("team_ref_finish", work["finish_position"].mean()),
        ("team_ref_points", work["points"].mean()),
        ("team_ref_pace", work["avg_lap_time_s"].mean()),
        ("driver_ref_win", work["win_probability"].mean()),
        ("driver_ref_finish", work["finish_position"].mean()),
        ("driver_ref_points", work["points"].mean()),
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(float(fallback))

    q = work["qualifying_position"].astype(float)
    q_score = -(q - q.mean()) / (q.std() + 1e-6)
    team_finish_score = -(work["team_ref_finish"] - work["team_ref_finish"].mean()) / (work["team_ref_finish"].std() + 1e-6)
    driver_finish_score = -(work["driver_ref_finish"] - work["driver_ref_finish"].mean()) / (work["driver_ref_finish"].std() + 1e-6)
    team_points_score = (work["team_ref_points"] - work["team_ref_points"].mean()) / (work["team_ref_points"].std() + 1e-6)
    driver_points_score = (work["driver_ref_points"] - work["driver_ref_points"].mean()) / (work["driver_ref_points"].std() + 1e-6)
    pace_score = -(work["team_ref_pace"] - work["team_ref_pace"].mean()) / (work["team_ref_pace"].std() + 1e-6)
    base_score = (work["win_probability"] - work["win_probability"].mean()) / (work["win_probability"].std() + 1e-6)

    work["_score"] = (
        0.26 * base_score
        + 0.24 * team_finish_score
        + 0.16 * driver_finish_score
        + 0.20 * team_points_score
        + 0.08 * driver_points_score
        + 0.04 * q_score
        + 0.02 * pace_score
    )

    # Winner-focused auxiliary term: emphasizes front-running team/driver form.
    work["_winner_focus"] = (
        0.45 * team_points_score
        + 0.25 * team_finish_score
        + 0.15 * driver_points_score
        + 0.10 * driver_finish_score
        + 0.05 * q_score
    )

    calibrated_parts = []
    for (_, _), g in work.groupby(["season", "race_round"], as_index=False):
        base = g["_score"].to_numpy(dtype=float)
        winner_focus = g["_winner_focus"].to_numpy(dtype=float)

        # Winner-focused optimization + top-end sharpening for P1-P3 separation.
        adjusted = (base + 0.28 * winner_focus) / 0.84
        raw_probs = _softmax(adjusted)

        multipliers = np.ones_like(raw_probs)
        top_idx = np.argsort(raw_probs)[::-1][:3]
        top_mult = np.array([1.34, 1.18, 1.08])
        for i, idx in enumerate(top_idx):
            multipliers[idx] = top_mult[i]

        sharp_probs = raw_probs * multipliers
        probs = 100 * sharp_probs / (sharp_probs.sum() + 1e-12)
        gg = g.copy()
        gg["win_probability"] = probs
        calibrated_parts.append(gg)

    work = pd.concat(calibrated_parts, ignore_index=True)

    # Confidence reflects prediction separation among front runners.
    work["confidence"] = 0.55
    for (_, _), g in work.groupby(["season", "race_round"], as_index=False):
        probs = np.sort(g["win_probability"].to_numpy())[::-1]
        gap = (probs[0] - probs[1]) if len(probs) > 1 else probs[0]
        conf = float(np.clip(0.55 + gap / 60.0, 0.5, 0.98))
        work.loc[g.index, "confidence"] = conf

    return work.drop(
        columns=[
            "_score",
            "team_ref_win",
            "team_ref_finish",
            "team_ref_points",
            "team_ref_pace",
            "driver_ref_win",
            "driver_ref_finish",
            "driver_ref_points",
            "_winner_focus",
        ],
        errors="ignore",
    )


def _build_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    seasons = [2024, 2025]
    races = [
        "Bahrain Grand Prix",
        "Saudi Arabian Grand Prix",
        "Australian Grand Prix",
        "Japanese Grand Prix",
        "Chinese Grand Prix",
    ]
    teams = {
        "VER": "Red Bull",
        "PER": "Red Bull",
        "LEC": "Ferrari",
        "SAI": "Ferrari",
        "HAM": "Mercedes",
        "RUS": "Mercedes",
        "NOR": "McLaren",
        "PIA": "McLaren",
        "ALO": "Aston Martin",
        "STR": "Aston Martin",
        "GAS": "Alpine",
        "OCO": "Alpine",
        "ALB": "Williams",
        "SAR": "Williams",
        "HUL": "Haas",
        "MAG": "Haas",
        "TSU": "RB",
        "LAW": "RB",
        "BOT": "Kick Sauber",
        "ZHO": "Kick Sauber",
    }

    base_strength = {
        "VER": 94,
        "PER": 88,
        "LEC": 87,
        "SAI": 85,
        "NOR": 84,
        "PIA": 82,
        "HAM": 81,
        "RUS": 80,
        "ALO": 77,
        "STR": 73,
        "GAS": 72,
        "OCO": 71,
        "TSU": 70,
        "LAW": 69,
        "ALB": 68,
        "SAR": 64,
        "HUL": 67,
        "MAG": 66,
        "BOT": 65,
        "ZHO": 63,
    }

    rows: list[dict] = []
    race_round = 0
    for season in seasons:
        for race in races:
            race_round += 1
            drivers = list(teams.keys())
            q_scores = {d: base_strength[d] + rng.normal(0, 5) for d in drivers}
            q_ranked = sorted(drivers, key=lambda d: q_scores[d], reverse=True)
            q_pos = {d: i + 1 for i, d in enumerate(q_ranked)}

            grid_noise = {d: rng.integers(-1, 2) for d in drivers}
            grid_pos = {d: int(np.clip(q_pos[d] + grid_noise[d], 1, 20)) for d in drivers}

            race_scores = {
                d: base_strength[d] + rng.normal(0, 7) - 0.8 * grid_pos[d] for d in drivers
            }
            finish_ranked = sorted(drivers, key=lambda d: race_scores[d], reverse=True)
            finish_pos = {d: i + 1 for i, d in enumerate(finish_ranked)}

            inv = np.array([1.0 / (finish_pos[d] + 0.75 * grid_pos[d]) for d in drivers])
            win_prob = 100 * inv / inv.sum()

            points_map = {
                1: 25,
                2: 18,
                3: 15,
                4: 12,
                5: 10,
                6: 8,
                7: 6,
                8: 4,
                9: 2,
                10: 1,
            }

            for idx, d in enumerate(drivers):
                fp = finish_pos[d]
                points = points_map.get(fp, 0)
                rows.append(
                    {
                        "season": season,
                        "race": race,
                        "race_round": race_round,
                        "driver": d,
                        "constructor": teams[d],
                        "qualifying_position": q_pos[d],
                        "grid_position": grid_pos[d],
                        "finish_position": fp,
                        "points": points,
                        "pit_stops": int(np.clip(2 + rng.integers(-1, 2), 1, 4)),
                        "avg_lap_time_s": 90.0 + 0.08 * fp + rng.normal(0, 0.35),
                        "team_performance": base_strength[d],
                        "historical_form": base_strength[d] + rng.normal(0, 3),
                        "win_probability": float(win_prob[idx]),
                        "confidence": float(np.clip(0.55 + 0.02 * (21 - grid_pos[d]) + rng.normal(0, 0.05), 0.35, 0.97)),
                    }
                )

    return pd.DataFrame(rows)


def _normalize_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    col_map = {
        "season": _find_col(df, ["season", "year"]),
        "race": _find_col(df, ["race", "race_name", "event", "event_name", "grand_prix"]),
        "driver": _find_col(df, ["driver", "driver_code", "driver_name"]),
        "driver_name": _find_col(df, ["driver_name", "name"]),
        "constructor": _find_col(df, ["constructor", "team", "team_name"]),
        "qualifying_position": _find_col(df, ["qualifying_position", "quali_position", "q_pos", "quali_pos"]),
        "grid_position": _find_col(df, ["grid_position", "grid", "start_position"]),
        "finish_position": _find_col(df, ["finish_position", "position", "race_position", "final_position"]),
        "points": _find_col(df, ["points", "race_points"]),
        "pit_stops": _find_col(df, ["pit_stops", "pitstop_count", "pit_count"]),
        "avg_lap_time_s": _find_col(df, ["avg_lap_time_s", "avg_lap_time", "lap_time", "mean_lap_time"]),
        "team_performance": _find_col(df, ["team_performance", "team_elo", "team_strength"]),
        "historical_form": _find_col(df, ["historical_form", "driver_elo", "driver_form"]),
        "win_probability": _find_col(df, ["win_probability", "win_prob", "prob_win", "prediction_prob"]),
        "confidence": _find_col(df, ["confidence", "confidence_score", "pred_confidence"]),
    }

    for target, source in col_map.items():
        if source is not None and source != target:
            df[target] = df[source]

    if "season" not in df.columns:
        df["season"] = 2025
    if "race" not in df.columns:
        df["race"] = "Unknown Grand Prix"
    if "driver" not in df.columns:
        df["driver"] = [f"DR{i+1:02d}" for i in range(len(df))]
    if "driver_name" not in df.columns:
        df["driver_name"] = df["driver"]
    if "constructor" not in df.columns:
        df["constructor"] = "Unknown"

    for col, fallback in [
        ("qualifying_position", 10),
        ("grid_position", 10),
        ("finish_position", 10),
        ("points", 0),
        ("pit_stops", 2),
        ("avg_lap_time_s", 91.5),
        ("team_performance", 75),
        ("historical_form", 75),
    ]:
        if col not in df.columns:
            df[col] = fallback
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fallback)

    if "win_probability" not in df.columns:
        inv = 1 / (df["finish_position"].astype(float) + 0.75 * df["grid_position"].astype(float))
        df["win_probability"] = 100 * inv / inv.sum()
    df["win_probability"] = pd.to_numeric(df["win_probability"], errors="coerce").fillna(0.0)

    if "confidence" not in df.columns:
        df["confidence"] = np.clip(0.5 + 0.015 * (21 - df["grid_position"]), 0.35, 0.95)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.5)

    if "race_round" not in df.columns:
        race_order = (
            df[["season", "race"]]
            .drop_duplicates()
            .sort_values(["season", "race"])
            .reset_index(drop=True)
        )
        race_order["race_round"] = race_order.groupby("season").cumcount() + 1
        df = df.merge(race_order, on=["season", "race"], how="left")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(2025).astype(int)
    df["race"] = df["race"].astype(str)
    df["driver"] = df["driver"].astype(str).replace(DRIVER_CODE_ALIAS)
    df["driver_name"] = df["driver_name"].astype(str)
    df["constructor"] = df["constructor"].astype(str).str.strip()
    df.loc[df["constructor"].str.lower().isin(["nan", "none", ""]), "constructor"] = "Unknown"

    return df


def load_dataset(path: Optional[Path] = None, prediction_season: int = DEFAULT_PREDICTION_SEASON) -> pd.DataFrame:
    if path and path.exists():
        p = pd.read_csv(path)
        if path.name.lower() == "predictions.csv":
            p = p.copy()
            yr, gp = _infer_race_meta_from_path(path)
            if gp and "race" not in p.columns:
                p["race"] = gp
            if "season" not in p.columns:
                p["season"] = prediction_season if prediction_season is not None else (yr or DEFAULT_PREDICTION_SEASON)
            if "driver_code" in p.columns and "driver" not in p.columns:
                p["driver"] = p["driver_code"].astype(str).str.upper()
            if "constructor" not in p.columns:
                lookup = _build_latest_driver_team_lookup()
                p["constructor"] = p.get("driver", pd.Series(dtype=str)).map(lookup)
                p["constructor"] = p["constructor"].fillna(p.get("driver", pd.Series(dtype=str)).map(DRIVER_TEAM_FALLBACK)).fillna("Unknown")
        return _normalize_dataset(p)

    pred_dataset = _load_predictions_from_processed_dirs(prediction_season)
    if pred_dataset is not None:
        return _normalize_dataset(pred_dataset)

    candidates = [
        PROC_DIR / "historical_results_2025_extended.csv",
        PROC_DIR / "historical_results.csv",
    ]

    found = _first_existing(candidates)
    if found is not None:
        return _normalize_dataset(pd.read_csv(found))

    return _normalize_dataset(_build_demo_data())


DATA = load_dataset(prediction_season=DEFAULT_PREDICTION_SEASON)


# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------
def _base_layout(title: str, height: int = 360) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, family="Titillium Web", color=THEME["text"])),
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        font=dict(color=THEME["text"], family="IBM Plex Sans", size=11),
        margin=dict(l=45, r=20, t=44, b=40),
        height=height,
        xaxis=dict(gridcolor=THEME["border"], zerolinecolor=THEME["border"]),
        yaxis=dict(gridcolor=THEME["border"], zerolinecolor=THEME["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )


def fig_win_prob(df: pd.DataFrame) -> go.Figure:
    race_df = (
        df.groupby(["driver", "driver_name", "constructor"], as_index=False)
        .agg(win_probability=("win_probability", "mean"), confidence=("confidence", "mean"))
        .sort_values("win_probability", ascending=False)
    )

    race_df = race_df.head(GRID_SIZE_DEFAULT)
    top3 = set(race_df.head(3)["driver"].tolist())
    colors = []
    line_colors = []
    line_widths = []
    for _, row in race_df.iterrows():
        colors.append(TEAM_COLORS.get(row["constructor"], "#5E697D"))
        if row["driver"] in top3:
            line_colors.append("#FFFFFF")
            line_widths.append(2.0)
        else:
            line_colors.append("rgba(0,0,0,0)")
            line_widths.append(0)

    fig = go.Figure(
        go.Bar(
            x=race_df["driver"],
            y=race_df["win_probability"],
            marker=dict(color=colors, line=dict(color=line_colors, width=line_widths)),
            text=[f"{v:.1f}%" for v in race_df["win_probability"]],
            textposition="outside",
            customdata=np.stack([race_df["driver_name"], race_df["constructor"], race_df["confidence"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{x})<br>"
                "Constructor: %{customdata[1]}<br>"
                "Win probability: %{y:.2f}%<br>"
                "Confidence: %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**_base_layout("Race Prediction Overview: Win Probabilities", height=380))
    fig.update_yaxes(title="Predicted win probability (%)")
    fig.update_xaxes(title="Driver")
    return fig


def fig_quali_finish_trend(df: pd.DataFrame) -> go.Figure:
    trend_df = (
        df.groupby(["race_round", "race", "driver"], as_index=False)
        .agg(
            qualifying_position=("qualifying_position", "mean"),
            finish_position=("finish_position", "mean"),
        )
        .sort_values("race_round")
    )

    if trend_df.empty:
        return go.Figure().update_layout(**_base_layout("Qualifying vs Finish Trend", 340))

    fig = go.Figure()
    for drv in trend_df["driver"].unique()[:6]:
        sub = trend_df[trend_df["driver"] == drv]
        fig.add_trace(
            go.Scatter(
                x=sub["race_round"],
                y=sub["qualifying_position"],
                mode="lines+markers",
                line=dict(width=2, dash="dot"),
                name=f"{drv} Quali",
                hovertemplate="Driver: " + drv + "<br>Round %{x}<br>Q Position: %{y:.0f}<extra></extra>",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub["race_round"],
                y=sub["finish_position"],
                mode="lines+markers",
                line=dict(width=2),
                name=f"{drv} Finish",
                hovertemplate="Driver: " + drv + "<br>Round %{x}<br>Finish: %{y:.0f}<extra></extra>",
                showlegend=True,
            )
        )

    fig.update_layout(**_base_layout("Driver Performance: Qualifying vs Finish", 380))
    fig.update_yaxes(title="Position (lower is better)", autorange="reversed")
    fig.update_xaxes(title="Season round")
    return fig


def fig_points_over_season(df: pd.DataFrame) -> go.Figure:
    points_df = (
        df.groupby(["season", "race_round", "driver"], as_index=False)
        .agg(points=("points", "sum"))
        .sort_values(["season", "race_round"])
    )
    points_df["cum_points"] = points_df.groupby(["season", "driver"])["points"].cumsum()

    fig = px.line(
        points_df,
        x="race_round",
        y="cum_points",
        color="driver",
        line_group="driver",
        markers=True,
        custom_data=["season"],
    )
    fig.update_traces(
        hovertemplate=(
            "Driver: %{legendgroup}<br>"
            "Season: %{customdata[0]}<br>"
            "Round: %{x}<br>"
            "Cumulative points: %{y}<extra></extra>"
        )
    )
    fig.update_layout(**_base_layout("Driver Performance: Points over Season", 360))
    fig.update_xaxes(title="Season round")
    fig.update_yaxes(title="Cumulative points")
    return fig


def fig_team_comparison(df: pd.DataFrame) -> go.Figure:
    team_df = (
        df.groupby(["constructor", "driver"], as_index=False)
        .agg(
            grid_position=("grid_position", "mean"),
            finish_position=("finish_position", "mean"),
            points=("points", "sum"),
        )
    )

    fig = px.scatter(
        team_df,
        x="grid_position",
        y="finish_position",
        color="constructor",
        size="points",
        hover_name="driver",
        color_discrete_map=TEAM_COLORS,
    )
    fig.update_traces(
        hovertemplate=(
            "Driver: %{hovertext}<br>"
            "Team: %{marker.color}<br>"
            "Avg grid: %{x:.2f}<br>"
            "Avg finish: %{y:.2f}<extra></extra>"
        )
    )
    fig.update_layout(**_base_layout("Driver Performance: Team Comparison", 360))
    fig.update_xaxes(title="Average grid position")
    fig.update_yaxes(title="Average finish position", autorange="reversed")
    return fig


def calc_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    target = "finish_position"
    candidates = [
        "qualifying_position",
        "grid_position",
        "pit_stops",
        "points",
        "avg_lap_time_s",
        "team_performance",
        "historical_form",
        "win_probability",
        "confidence",
        "race_round",
    ]
    use = [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if target not in use:
        use.append(target)

    corr = df[use].corr(numeric_only=True)
    if target not in corr.columns:
        imp = pd.DataFrame({"feature": ["qualifying_position"], "importance": [1.0]})
    else:
        imp = (
            corr[target]
            .drop(labels=[target], errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        imp.columns = ["feature", "importance"]

    if imp.empty:
        imp = pd.DataFrame(
            {
                "feature": [
                    "qualifying_position",
                    "grid_position",
                    "team_performance",
                    "historical_form",
                    "pit_stops",
                ],
                "importance": [0.92, 0.88, 0.74, 0.69, 0.45],
            }
        )
    return imp


def fig_feature_importance(df: pd.DataFrame) -> go.Figure:
    imp = calc_feature_importance(df)
    fig = go.Figure(
        go.Bar(
            x=imp["importance"],
            y=imp["feature"],
            orientation="h",
            marker=dict(color=THEME["accent"]),
            text=[f"{v:.2f}" for v in imp["importance"]],
            textposition="outside",
            hovertemplate="Feature: %{y}<br>Importance: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Feature Importance / Insights", 360))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Absolute impact score")
    return fig


def fig_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] < 2:
        return go.Figure().update_layout(**_base_layout("Correlation Heatmap", 340))

    corr = numeric_df.corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            colorbar=dict(title="corr"),
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Advanced Insights: Correlation Heatmap", 380))
    return fig


def fig_grid_finish(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="grid_position",
        y="finish_position",
        color="constructor",
        hover_name="driver",
        color_discrete_map=TEAM_COLORS,
        opacity=0.8,
    )
    fig.update_layout(**_base_layout("Grid Position vs Finish Position", 350))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Grid position")
    fig.update_yaxes(title="Finish position")
    return fig


def fig_pit_outcome(df: pd.DataFrame) -> go.Figure:
    outcome = (
        df.groupby("pit_stops", as_index=False)
        .agg(avg_finish=("finish_position", "mean"), avg_points=("points", "mean"))
        .sort_values("pit_stops")
    )

    fig = go.Figure(
        go.Scatter(
            x=outcome["pit_stops"],
            y=outcome["avg_finish"],
            mode="lines+markers",
            marker=dict(size=10, color=THEME["accent"]),
            line=dict(width=3, color=THEME["accent_soft"]),
            customdata=np.stack([outcome["avg_points"]], axis=-1),
            hovertemplate=(
                "Pit stops: %{x}<br>"
                "Average finish: %{y:.2f}<br>"
                "Average points: %{customdata[0]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**_base_layout("Pit Stops vs Race Outcome", 350))
    fig.update_yaxes(title="Average finish position", autorange="reversed")
    fig.update_xaxes(title="Pit stop count", dtick=1)
    return fig


def fig_what_if(df: pd.DataFrame, driver: str, new_quali_pos: int) -> go.Figure:
    driver_row = (
        df[df["driver"] == driver]
        .sort_values(["season", "race_round"], ascending=[False, False])
        .head(1)
    )

    if driver_row.empty:
        baseline = 5.0
        current_quali = 10
    else:
        baseline = float(driver_row["win_probability"].iloc[0])
        current_quali = int(driver_row["qualifying_position"].iloc[0])

    delta = (current_quali - new_quali_pos) * 0.8
    adjusted = float(np.clip(baseline + delta, 0.1, 95.0))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Current", "What-if"],
            y=[baseline, adjusted],
            marker=dict(color=["#5A6478", THEME["accent"]]),
            text=[f"{baseline:.1f}%", f"{adjusted:.1f}%"],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("What-if Analysis: Qualifying Position Impact", 320))
    fig.update_yaxes(title="Predicted win probability (%)")
    fig.update_xaxes(title="Scenario")
    return fig


def build_story_text(df: pd.DataFrame) -> str:
    top3 = (df["grid_position"] <= 3)
    if top3.any() and (~top3).any():
        top3_win = df.loc[top3, "win_probability"].mean()
        rest_win = df.loc[~top3, "win_probability"].mean()
        if rest_win > 0:
            boost = ((top3_win - rest_win) / rest_win) * 100
        else:
            boost = 0
    else:
        boost = 0

    return (
        f"Drivers starting in top 3 have {boost:.0f}% higher win probability. "
        "Qualifying and grid position continue to dominate race outcome forecasts."
    )


def _filter_df(
    df: pd.DataFrame,
    season: Optional[int],
    race: Optional[str],
    drivers: Optional[list[str]],
    constructor: Optional[str],
) -> pd.DataFrame:
    out = df.copy()
    if season not in (None, "All"):
        out = out[out["season"] == int(season)]
    if race not in (None, "All"):
        out = out[out["race"] == race]
    if isinstance(drivers, str):
        drivers = [drivers]
    if drivers and "All" not in drivers:
        out = out[out["driver"].isin(drivers)]
    if constructor not in (None, "All"):
        out = out[out["constructor"] == constructor]
    return out


def _apply_quali_overrides(df: pd.DataFrame, quali_rows: Optional[list[dict]]) -> pd.DataFrame:
    """Apply editable qualifying inputs and re-predict race win probabilities."""
    if df.empty or not quali_rows:
        return df

    overrides: dict[str, int] = {}
    for row in quali_rows:
        driver = str(row.get("driver", "")).strip().upper()
        if not driver:
            continue
        q_raw = pd.to_numeric(row.get("qualifying_position"), errors="coerce")
        if pd.isna(q_raw):
            continue
        q = int(np.clip(float(q_raw), 1, GRID_SIZE_DEFAULT))
        overrides[driver] = q

    if not overrides:
        return df

    work = df.copy()
    work["driver"] = work["driver"].astype(str).str.upper()
    work["qualifying_position"] = work["driver"].map(overrides).fillna(work["qualifying_position"])
    work["grid_position"] = work["qualifying_position"]

    recalibrated_parts = []
    for (_, _), g in work.groupby(["season", "race_round"], as_index=False):
        gg = g.copy()
        base = pd.to_numeric(gg["win_probability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        q = pd.to_numeric(gg["qualifying_position"], errors="coerce").fillna(GRID_SIZE_DEFAULT).to_numpy(dtype=float)

        base_score = (base - base.mean()) / (base.std() + 1e-6)
        q_score = -(q - q.mean()) / (q.std() + 1e-6)

        # For race-day use, weigh known qualifying heavily while preserving model priors.
        score = 0.55 * base_score + 0.45 * q_score
        probs = 100 * _softmax(score)

        gg["win_probability"] = probs
        sorted_probs = np.sort(probs)[::-1]
        gap = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else sorted_probs[0]
        conf = float(np.clip(0.55 + gap / 55.0, 0.5, 0.99))
        gg["confidence"] = conf
        recalibrated_parts.append(gg)

    return pd.concat(recalibrated_parts, ignore_index=True)


# -----------------------------------------------------------------------------
# Dash app
# -----------------------------------------------------------------------------
def create_app(predictions_path: Optional[Path] = None) -> dash.Dash:
    global DATA
    if predictions_path:
        DATA = load_dataset(predictions_path)

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="F1 Race Prediction Dashboard",
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    season_data = DATA.loc[DATA["season"] == DEFAULT_PREDICTION_SEASON].copy()
    if season_data.empty:
        season_data = DATA.copy()

    races = sorted(season_data["race"].unique().tolist())
    default_race = "Australian Grand Prix" if "Australian Grand Prix" in races else (races[0] if races else None)
    drivers = sorted(DATA["driver"].unique().tolist())

    app.index_string = """
    <!DOCTYPE html>
    <html>
      <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@600;700&family=Inter:wght@400;500;600&display=swap');

          :root {
            --bg: #0F1218;
            --panel: #171C24;
            --panel-alt: #1D2430;
            --border: #2C3443;
            --text: #F5F7FA;
            --muted: #9CA8BC;
            --accent: #E10600;
          }

          * { box-sizing: border-box; }

          body {
            margin: 0;
            background: linear-gradient(180deg, rgba(225,6,0,0.06), transparent 18%), var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
          }

          .page-wrap {
            max-width: 1600px;
            margin: 0 auto;
            padding: 18px 16px 28px;
          }

          .header-card,
          .section-card,
          .kpi-card {
            background: linear-gradient(180deg, var(--panel-alt), var(--panel));
            border: 1px solid var(--border);
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
          }

          .header-card {
            padding: 14px;
            margin-bottom: 14px;
          }

          .title {
            font-family: 'Titillium Web', sans-serif;
            font-size: 30px;
            font-weight: 700;
            margin: 0;
            letter-spacing: 0.2px;
          }

          .subtitle {
            margin-top: 2px;
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.1px;
            font-weight: 600;
          }

          .filters-grid {
            margin-top: 14px;
          }

                    .season-badge-wrap {
                        display: flex;
                        justify-content: flex-end;
                        align-items: flex-end;
                        height: 100%;
                    }

                    .season-badge {
                        font-family: 'Titillium Web', sans-serif;
                        font-size: 20px;
                        font-weight: 700;
                        letter-spacing: 1px;
                        color: #FFFFFF;
                        padding: 8px 14px;
                        border-radius: 999px;
                        border: 1px solid var(--border);
                        background: linear-gradient(90deg, rgba(225,6,0,0.85), rgba(225,6,0,0.5));
                        box-shadow: 0 4px 16px rgba(225, 6, 0, 0.28);
                    }

          .filter-label {
            font-size: 10px;
            text-transform: uppercase;
            color: var(--muted);
            letter-spacing: 1px;
            font-weight: 700;
            margin-bottom: 5px;
          }

          .kpi-row {
            margin-bottom: 14px;
          }

          .kpi-card {
            padding: 12px 14px;
            border-top: 3px solid var(--accent);
          }

          .kpi-title {
            font-size: 10px;
            text-transform: uppercase;
            color: var(--muted);
            letter-spacing: 1.1px;
            font-weight: 700;
          }

          .kpi-value {
            font-family: 'Titillium Web', sans-serif;
            font-size: 28px;
            font-weight: 700;
            line-height: 1.1;
          }

          .kpi-sub {
            color: var(--muted);
            font-size: 12px;
          }

          .section-card {
            padding: 12px;
            margin-bottom: 14px;
          }

          .section-title {
            font-family: 'Titillium Web', sans-serif;
            font-size: 18px;
            margin: 0 0 8px;
          }

                    .tab-shell {
                        margin-bottom: 14px;
                    }

                    .tab-label {
                        font-family: 'Titillium Web', sans-serif;
                        letter-spacing: 0.3px;
                    }

          .insight-text {
            color: var(--muted);
            font-size: 13px;
            margin: 8px 0 0;
          }

          .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
            --accent: #E10600;
          }

          .tooltip-target {
            display: inline-block;
            margin-left: 6px;
            color: #FF8E8E;
            cursor: pointer;
            font-weight: 700;
          }

                    /* Dash dcc.Dropdown text visibility fix */
                    .Select-control {
                        background: #FFFFFF !important;
                        border: 1px solid #D1D7E0 !important;
                    }

                    .Select-placeholder,
                    .Select-value-label,
                    .Select-input > input,
                    .Select-value,
                    .has-value.Select--single > .Select-control .Select-value,
                    .has-value.Select--single > .Select-control .Select-value .Select-value-label,
                    .Select--single > .Select-control .Select-value,
                    .Select--single > .Select-control .Select-value .Select-value-label,
                    .is-focused:not(.is-open) > .Select-control .Select-value,
                    .is-focused:not(.is-open) > .Select-control .Select-value .Select-value-label {
                        color: #111111 !important;
                    }

                    .Select-menu-outer,
                    .Select-menu {
                        background: #FFFFFF !important;
                        border: 1px solid #D1D7E0 !important;
                    }

                    .Select-option {
                        color: #111111 !important;
                        background: #FFFFFF !important;
                    }

                    .Select-option.is-focused {
                        color: #111111 !important;
                        background: #EEF2F7 !important;
                    }

                    .Select-option.is-selected {
                        color: #111111 !important;
                        background: #E3EAF5 !important;
                    }

                    .Select--multi .Select-value {
                        background: #F0F3F8 !important;
                        border-color: #D1D7E0 !important;
                    }

                    .Select--multi .Select-value-label,
                    .Select--multi .Select-value-icon {
                        color: #111111 !important;
                    }

                    .Select-arrow,
                    .Select-clear-zone {
                        color: #111111 !important;
                    }

                    /* Fallback for virtualized dropdown rendering */
                    .Select * {
                        color: #111111 !important;
                    }

                    .VirtualizedSelectOption,
                    .VirtualizedSelectFocusedOption {
                        color: #111111 !important;
                        background: #FFFFFF !important;
                    }

                    .VirtualizedSelectFocusedOption {
                        background: #EEF2F7 !important;
                    }

          footer {
            text-align: center;
            color: var(--muted);
            font-size: 11px;
            margin-top: 8px;
          }
        </style>
      </head>
      <body>
        {%app_entry%}
                <footer>F1 Prediction Dashboard</footer>
        {%config%}
        {%scripts%}
        {%renderer%}
      </body>
    </html>
    """

    app.layout = html.Div(
        className="page-wrap",
        children=[
            html.Div(
                className="header-card",
                children=[
                    html.H1("Formula 1 Race Prediction Dashboard", className="title"),
                    dbc.Row(
                        className="filters-grid",
                        children=[
                            dbc.Col(
                                [
                                    html.Div("Race", className="filter-label"),
                                    dcc.Dropdown(
                                        id="race-filter",
                                        options=[{"label": r, "value": r} for r in races],
                                        value=default_race,
                                        clearable=False,
                                        style={"backgroundColor": "#FFFFFF", "color": "#111111"},
                                    ),
                                ],
                                md=8,
                            ),
                            dbc.Col(
                                [
                                    html.Div(className="season-badge-wrap", children=[
                                        html.Div("Season 2026", className="season-badge")
                                    ])
                                ],
                                md=4,
                            ),
                        ],
                    ),
                ],
            ),

            dbc.Row(
                className="kpi-row",
                children=[
                    dbc.Col(
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.Div("Most probable winner", className="kpi-title"),
                                html.Div(id="kpi-winner", className="kpi-value"),
                                html.Div(id="kpi-winner-sub", className="kpi-sub"),
                            ],
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.Div("Confidence score", className="kpi-title"),
                                html.Div(id="kpi-confidence", className="kpi-value"),
                                html.Div("Average confidence of top prediction", className="kpi-sub"),
                            ],
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        html.Div(
                            className="kpi-card",
                            children=[
                                html.Div("Total drivers", className="kpi-title"),
                                html.Div(id="kpi-total-drivers", className="kpi-value"),
                                html.Div("Drivers in current filter context", className="kpi-sub"),
                            ],
                        ),
                        md=4,
                    ),
                ],
            ),

            dcc.Tabs(
                className="tab-shell",
                children=[
                    dcc.Tab(
                        label="Race Overview",
                        className="tab-label",
                        children=[
                            html.Div(
                                className="section-card",
                                children=[
                                    html.H3("Tomorrow Race Input: Editable Qualifying", className="section-title"),
                                    html.Div(
                                        "Enter/adjust qualifying positions below for the selected race. Predictions update automatically.",
                                        className="insight-text",
                                    ),
                                    dash_table.DataTable(
                                        id="quali-edit-table",
                                        columns=[
                                            {"name": "Driver", "id": "driver", "editable": False},
                                            {"name": "Driver Name", "id": "driver_name", "editable": False},
                                            {"name": "Constructor", "id": "constructor", "editable": False},
                                            {"name": "Quali Pos", "id": "qualifying_position", "editable": True, "type": "numeric"},
                                        ],
                                        data=[],
                                        editable=True,
                                        page_action="none",
                                        style_table={"maxHeight": "420px", "overflowY": "auto", "border": f"1px solid {THEME['border']}"},
                                        style_header={
                                            "backgroundColor": THEME["panel_alt"],
                                            "color": THEME["text"],
                                            "fontWeight": "700",
                                            "border": f"1px solid {THEME['border']}",
                                        },
                                        style_data={
                                            "backgroundColor": THEME["panel"],
                                            "color": THEME["text"],
                                            "border": f"1px solid {THEME['border']}",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                className="section-card",
                                children=[
                                    html.H3("Race Prediction Overview", className="section-title"),
                                    dcc.Graph(id="win-prob-chart", config={"displayModeBar": False}),
                                    dash_table.DataTable(
                                        id="prediction-table",
                                        style_as_list_view=True,
                                        style_table={"overflowX": "auto", "border": f"1px solid {THEME['border']}"},
                                        style_header={
                                            "backgroundColor": THEME["panel_alt"],
                                            "color": THEME["text"],
                                            "fontWeight": "700",
                                            "border": f"1px solid {THEME['border']}",
                                        },
                                        style_data={
                                            "backgroundColor": THEME["panel"],
                                            "color": THEME["text"],
                                            "border": f"1px solid {THEME['border']}",
                                        },
                                        page_size=8,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Driver Analysis",
                        className="tab-label",
                        children=[
                            html.Div(
                                className="section-card",
                                children=[
                                    html.H3("Driver Performance Analysis", className="section-title"),
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id="quali-finish-chart", config={"displayModeBar": False}), md=6),
                                            dbc.Col(dcc.Graph(id="points-line-chart", config={"displayModeBar": False}), md=6),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Team Performance",
                        className="tab-label",
                        children=[
                            html.Div(
                                className="section-card",
                                children=[
                                    html.H3("Team Performance", className="section-title"),
                                    dcc.Graph(id="team-comparison-chart", config={"displayModeBar": False}),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Insights",
                        className="tab-label",
                        children=[
                            html.Div(
                                className="section-card",
                                children=[
                                    html.H3("Feature Importance / Insights", className="section-title"),
                                    html.Div(
                                        [
                                            "Model drivers of race outcome",
                                            html.Span("i", id="fi-tooltip-target", className="tooltip-target"),
                                            dbc.Tooltip(
                                                "Qualifying position has highest impact on race outcome",
                                                target="fi-tooltip-target",
                                                placement="right",
                                            ),
                                        ],
                                        className="insight-text",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id="feature-importance-chart", config={"displayModeBar": False}), md=6),
                                            dbc.Col(dcc.Graph(id="corr-heatmap-chart", config={"displayModeBar": False}), md=6),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id="grid-finish-chart", config={"displayModeBar": False}), md=6),
                                            dbc.Col(dcc.Graph(id="pit-outcome-chart", config={"displayModeBar": False}), md=6),
                                        ]
                                    ),
                                    html.Div(id="story-text", className="insight-text"),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="What-if",
                        className="tab-label",
                        children=[
                            html.Div(
                                className="section-card",
                                children=[
                                    html.H3("What-if Analysis", className="section-title"),
                                    html.Div(
                                        "Adjust qualifying position and estimate impact on win probability for a selected driver.",
                                        className="insight-text",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Div("Driver", className="filter-label"),
                                                    dcc.Dropdown(
                                                        id="whatif-driver",
                                                        options=[{"label": d, "value": d} for d in drivers],
                                                        value=drivers[0] if drivers else None,
                                                        clearable=False,
                                                        style={"backgroundColor": "#FFFFFF", "color": "#111111"},
                                                    ),
                                                ],
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Div("Qualifying position", className="filter-label"),
                                                    dcc.Slider(
                                                        id="whatif-quali",
                                                        min=1,
                                                        max=GRID_SIZE_DEFAULT,
                                                        step=1,
                                                        value=5,
                                                        marks={i: str(i) for i in [1, 5, 10, 15, GRID_SIZE_DEFAULT]},
                                                    ),
                                                ],
                                                md=8,
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id="whatif-chart", config={"displayModeBar": False}), md=8),
                                            dbc.Col(html.Div(id="whatif-text", className="insight-text"), md=4),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @callback(
        Output("quali-edit-table", "data"),
        Input("race-filter", "value"),
    )
    def update_quali_table(race):
        subset = _filter_df(DATA, DEFAULT_PREDICTION_SEASON, race, None, None)
        if subset.empty:
            return []

        table_df = (
            subset.groupby(["driver", "driver_name", "constructor"], as_index=False)
            .agg(
                qualifying_position=("qualifying_position", "mean"),
                win_probability=("win_probability", "mean"),
            )
            .sort_values("win_probability", ascending=False)
            .drop(columns=["win_probability"])
        )
        table_df["qualifying_position"] = (
            pd.to_numeric(table_df["qualifying_position"], errors="coerce")
            .fillna(GRID_SIZE_DEFAULT)
            .round()
            .astype(int)
            .clip(1, GRID_SIZE_DEFAULT)
        )
        return table_df.to_dict("records")

    @callback(
        Output("whatif-driver", "options"),
        Output("whatif-driver", "value"),
        Input("race-filter", "value"),
        State("whatif-driver", "value"),
    )
    def update_driver_options(race, current_whatif):
        subset = _filter_df(DATA, DEFAULT_PREDICTION_SEASON, race, None, None)
        driver_labels = (
            subset[["driver", "driver_name"]]
            .drop_duplicates()
            .sort_values("driver")
        )
        available = driver_labels["driver"].tolist()
        whatif_options = [
            {"label": f"{r.driver_name} ({r.driver})", "value": r.driver}
            for r in driver_labels.itertuples(index=False)
        ]

        if current_whatif not in available:
            current_whatif = available[0] if available else None

        return whatif_options, current_whatif

    @callback(
        Output("kpi-winner", "children"),
        Output("kpi-winner-sub", "children"),
        Output("kpi-confidence", "children"),
        Output("kpi-total-drivers", "children"),
        Output("win-prob-chart", "figure"),
        Output("prediction-table", "columns"),
        Output("prediction-table", "data"),
        Output("quali-finish-chart", "figure"),
        Output("points-line-chart", "figure"),
        Output("team-comparison-chart", "figure"),
        Output("feature-importance-chart", "figure"),
        Output("corr-heatmap-chart", "figure"),
        Output("grid-finish-chart", "figure"),
        Output("pit-outcome-chart", "figure"),
        Output("story-text", "children"),
        Output("whatif-chart", "figure"),
        Output("whatif-text", "children"),
        Input("race-filter", "value"),
        Input("quali-edit-table", "data"),
        Input("whatif-driver", "value"),
        Input("whatif-quali", "value"),
    )
    def update_dashboard(
        race,
        quali_rows,
        whatif_driver,
        whatif_quali,
    ):
        filtered = _filter_df(DATA, DEFAULT_PREDICTION_SEASON, race, None, None)
        filtered = _apply_quali_overrides(filtered, quali_rows)
        if filtered.empty:
            filtered = DATA.copy()

        race_view = (
            filtered.groupby(["driver", "driver_name", "constructor"], as_index=False)
            .agg(
                win_probability=("win_probability", "mean"),
                confidence=("confidence", "mean"),
                qualifying_position=("qualifying_position", "mean"),
                finish_position=("finish_position", "mean"),
            )
            .sort_values("win_probability", ascending=False)
        )

        winner = race_view.iloc[0]["driver_name"] if not race_view.empty else "N/A"
        winner_team = race_view.iloc[0]["constructor"] if not race_view.empty else "N/A"
        winner_prob = race_view.iloc[0]["win_probability"] if not race_view.empty else 0
        winner_conf = race_view.iloc[0]["confidence"] if not race_view.empty else 0

        kpi_winner = winner
        kpi_winner_sub = f"{winner_team} | Predicted win probability: {winner_prob:.1f}%"
        kpi_conf = f"{winner_conf:.2f}"
        kpi_total = str(race_view["driver"].nunique())

        table_cols = [
            {"name": "Driver", "id": "driver"},
            {"name": "Driver Name", "id": "driver_name"},
            {"name": "Constructor", "id": "constructor"},
            {"name": "Win %", "id": "win_probability"},
            {"name": "Confidence", "id": "confidence"},
            {"name": "Avg Q", "id": "qualifying_position"},
            {"name": "Avg Finish", "id": "finish_position"},
        ]

        table_data = race_view.head(GRID_SIZE_DEFAULT).copy()
        for c in ["win_probability", "confidence", "qualifying_position", "finish_position"]:
            table_data[c] = table_data[c].round(2)

        whatif_driver_value = whatif_driver or (race_view.iloc[0]["driver"] if not race_view.empty else "")
        whatif_fig = fig_what_if(filtered, whatif_driver_value, int(whatif_quali))

        driver_row = (
            filtered[filtered["driver"] == whatif_driver_value]
            .sort_values(["season", "race_round"], ascending=[False, False])
            .head(1)
        )
        if driver_row.empty:
            whatif_msg = "Select a driver to run scenario analysis."
        else:
            dname = str(driver_row["driver_name"].iloc[0]) if "driver_name" in driver_row.columns else whatif_driver_value
            dteam = str(driver_row["constructor"].iloc[0]) if "constructor" in driver_row.columns else "Unknown"
            current_q = int(driver_row["qualifying_position"].iloc[0])
            base_p = float(driver_row["win_probability"].iloc[0])
            adjusted = float(np.clip(base_p + (current_q - int(whatif_quali)) * 0.8, 0.1, 95.0))
            delta = adjusted - base_p
            direction = "increase" if delta >= 0 else "decrease"
            impact = "high" if abs(delta) >= 4 else "moderate" if abs(delta) >= 2 else "low"
            whatif_msg = (
                f"Scenario for {dname} ({whatif_driver_value}) - {dteam}: "
                f"moving qualifying from P{current_q} to P{int(whatif_quali)} gives an estimated "
                f"win probability of {adjusted:.1f}% ({direction} of {abs(delta):.1f} pts from {base_p:.1f}%). "
                f"Expected impact level: {impact}."
            )

        return (
            kpi_winner,
            kpi_winner_sub,
            kpi_conf,
            kpi_total,
            fig_win_prob(filtered),
            table_cols,
            table_data.to_dict("records"),
            fig_quali_finish_trend(filtered),
            fig_points_over_season(filtered),
            fig_team_comparison(filtered),
            fig_feature_importance(filtered),
            fig_corr_heatmap(filtered),
            fig_grid_finish(filtered),
            fig_pit_outcome(filtered),
            build_story_text(filtered),
            whatif_fig,
            whatif_msg,
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Race Prediction Dashboard")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions CSV")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pred_path = Path(args.predictions) if args.predictions else None
    app = create_app(predictions_path=pred_path)

    print("\nF1 Race Prediction Dashboard")
    print(f"Open: http://localhost:{args.port}\n")
    app.run(debug=args.debug, port=args.port, host="0.0.0.0")
