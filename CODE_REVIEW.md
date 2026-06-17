# F1 Predictor — Comprehensive Code Review

**Review Date**: March 2026  
**Scope**: All Python files in `src/`, `config/`, and requirements  
**Overall Assessment**: Well-structured project with solid architecture, but several critical error-handling gaps and architectural improvements needed.

---

## 🔴 CRITICAL ISSUES

### 1. Missing Error Handling in OpenF1 Client Network Calls
**File**: [src/ingestion/openf1_client.py](src/ingestion/openf1_client.py#L40)  
**Issue**: The `get()` and `get_sync()` methods don't handle connection timeouts or network errors gracefully.

```python
async def get(self, endpoint: str, **params: Any) -> list[dict]:
    # No try-catch for connection errors
    async with httpx.AsyncClient(timeout=self.timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()  # Raises, but no catch block in caller
        return resp.json()
```

**Why it matters**: Network failures during race-critical predictions will crash the pipeline with no fallback.

**How to fix**:
```python
async def get(self, endpoint: str, **params: Any) -> list[dict]:
    """GET /v1/{endpoint}?key=val&... → list of result dicts or empty if error."""
    try:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        log.error("Timeout fetching %s — returning empty list", endpoint)
        return []
    except httpx.HTTPError as e:
        log.error("HTTP error fetching %s: %s — returning empty list", endpoint, e)
        return []
    except Exception as e:
        log.error("Unexpected error fetching %s: %s — returning empty list", endpoint, e)
        return []
```

Also update callers to handle empty results:
```python
def get_compound_strategy_summary(self, session_key: int) -> pd.DataFrame:
    stints = self.get_stints(session_key)
    if stints.empty:
        log.warning("No stints data — returning empty DataFrame")
        return pd.DataFrame()
```

---

### 2. No Input Validation in Feature Engineering Functions
**File**: [src/features/engineer.py](src/features/engineer.py#L350)  
**Issue**: `build_feature_matrix()` doesn't validate that required columns exist in input DataFrames.

```python
def build_feature_matrix(
    raw_df: pd.DataFrame,
    historical_results: Optional[pd.DataFrame] = None,
    fit: bool = True,
    artifacts: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    # No validation of raw_df structure or columns
    # Later functions assume columns exist -> KeyError if missing
    df = raw_df.copy()
    if historical_results is not None:
        driver_elo, team_elo = build_elo_ratings(historical_results)
        # Assumes driver_code, team_name, circuit_id columns exist
        df["driver_elo"] = df["driver_code"].map(driver_elo.ratings).fillna(1500.0)
```

**Why it matters**: Silent data corruption if columns are missing. Error messages will be cryptic.

**How to fix**:
```python
def build_feature_matrix(
    raw_df: pd.DataFrame,
    historical_results: Optional[pd.DataFrame] = None,
    fit: bool = True,
    artifacts: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    # Validate inputs
    if raw_df.empty:
        raise ValueError("raw_df cannot be empty")
    
    required_cols = ["driver_code"]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df missing required columns: {missing}")
    
    if historical_results is not None:
        for col in ["driver_code", "team_name", "finish_position"]:
            if col not in historical_results.columns:
                raise ValueError(f"historical_results missing column: {col}")
    
    # ... rest of function
```

---

### 3. Incomplete Model Artifact Persistence
**File**: [src/models/ensemble.py](src/models/ensemble.py#L380) + [src/predict.py](src/predict.py#L140)  
**Issue**: When using trained models at inference, the encoders/scaler artifacts are reconstructed from `ALL_FEATURES` but may not match what was used during training if features change.

```python
# Training saves entire ensemble object (model + artifacts)
def save(self, path: Optional[Path] = None) -> Path:
    with open(path, "wb") as f:
        pickle.dump(self, f)  # Pickles the whole object including artifacts

# But at inference, feature engineering reconstructs encoder/scaler from config
def build_feature_matrix(...):
    # Uses encoders from artifacts dict if provided
    # But predict.py doesn't pass artifacts from loaded model
    artifacts = {}  # New dict, no loaded artifacts
```

**Why it matters**: If someone adds a categorical feature to `ALL_FEATURES` after training, inference will encode differently than training, causing prediction errors.

**How to fix**:
```python
# In ensemble.py - save all artifacts needed for inference
def save_artifacts(self, path: Optional[Path] = None) -> Path:
    """Save ensemble + associated preprocessing artifacts."""
    artifact_dict = {
        "model": self,
        "feature_cols": self.feature_cols,
        "categorical_cols": ["driver_code", "team_name", "circuit_id"],
    }
    path = path or MODEL_DIR / "ensemble_with_artifacts.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact_dict, f)
    return path

@classmethod
def load_with_artifacts(cls, path: Optional[Path] = None) -> tuple["F1StackingEnsemble", dict]:
    path = path or MODEL_DIR / "ensemble_with_artifacts.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj["model"], {"feature_cols": obj["feature_cols"]}
    return obj, {}  # Old format fallback

# In predict.py
ensemble = F1StackingEnsemble.load(model_path)
feature_matrix, artifacts = build_feature_matrix(
    feature_df,
    historical_results=historical,
    fit=False,
    artifacts=getattr(ensemble, "_artifacts", {}),  # Pass saved artifacts
)
```

---

### 4. Potential Data Leakage in Feature Engineering
**File**: [src/features/engineer.py](src/features/engineer.py#L88)  
**Issue**: When fitting ELO ratings and circuit affinity during training, the code doesn't prevent including the current race in historical data.

```python
def build_elo_ratings(historical_results: pd.DataFrame) -> tuple[EloRating, EloRating]:
    driver_elo = EloRating()
    team_elo   = EloRating()
    
    # Updates ELO from ALL races including current one
    for (year, gp), group in historical_results.groupby(["year", "gp"]):
        driver_elo.update_from_race(group, entity_col="driver_code")
        # Problem: if historical_results contains the race being trained,
        # the model sees the outcome before predicting
```

**Why it matters**: Model performance metrics will be artificially inflated.

**How to fix**:
```python
def build_elo_ratings(
    historical_results: pd.DataFrame,
    exclude_race: Optional[tuple[int, str]] = None,  # (year, gp)
) -> tuple[EloRating, EloRating]:
    driver_elo = EloRating()
    team_elo   = EloRating()
    
    # Filter out current race
    data = historical_results
    if exclude_race is not None:
        year, gp = exclude_race
        data = data[(data["year"] != year) | (data["gp"] != gp)]
    
    for (year, gp), group in data.groupby(["year", "gp"]):
        driver_elo.update_from_race(group, entity_col="driver_code")
        team_elo.update_from_race(group, entity_col="team_name")
    
    return driver_elo, team_elo
```

---

### 5. Missing Validation of Historical CSV in Training
**File**: [src/train.py](src/train.py#L38)  
**Issue**: `load_historical()` checks for required columns but doesn't verify data quality or shape.

```python
def load_historical(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows × %d cols from %s", *df.shape, csv_path)
    required = ["year", "gp", "driver_code", "finish_position"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df  # No check for: duplicates, nulls in key cols, data types, etc.
```

**Why it matters**: Bad training data produces garbage models with no warning.

**How to fix**:
```python
def load_historical(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows × %d cols from %s", *df.shape, csv_path)
    
    # Check required columns
    required = ["year", "gp", "driver_code", "finish_position"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Validate no nulls in critical columns
    for col in required:
        nulls = df[col].isna().sum()
        if nulls > 0:
            log.warning("%s: %d null values (%.1f%%)", col, nulls, 
                       100 * nulls / len(df))
    
    # Validate finish_position is numeric and in valid range
    if not pd.api.types.is_numeric_dtype(df["finish_position"]):
        df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    invalid = (df["finish_position"] < 1) | (df["finish_position"] > 30)
    if invalid.sum() > 0:
        log.warning("Invalid finish_position values: %d rows", invalid.sum())
        df = df[~invalid]
    
    # Minimum viability check
    min_races = 5
    unique_races = df.groupby(["year", "gp"]).size()
    if len(unique_races) < min_races:
        raise ValueError(f"Need at least {min_races} races, got {len(unique_races)}")
    
    log.info("Validation complete: %d races, %d drivers", 
             len(unique_races), df["driver_code"].nunique())
    return df
```

---

### 6. PyTorch Model Fallback Doesn't Preserve State
**File**: [src/models/ensemble.py](src/models/ensemble.py#L125)  
**Issue**: `TireDegNN.predict()` checks `hasattr(self.model, "forward")` to detect PyTorch vs LightGBM, but if model is unpickled on a system without PyTorch, this fails.

```python
def predict(self, X: pd.DataFrame) -> np.ndarray:
    if not self._use_torch or not hasattr(self.model, "forward"):
        return self.model.predict(X)  # Falls back to LightGBM
    # But on PyTorch systems, self.model is a torch.nn.Module, not a pickle-able state
```

**Why it matters**: Models trained with PyTorch may not be loadable on inference boxes without PyTorch installed.

**How to fix**:
```python
def fit(self, X: pd.DataFrame, y: pd.Series) -> "TireDegNN":
    # ... training code ...
    # Always save model state as a LightGBM fallback
    self._fallback_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    self._fallback_model.fit(X, y)
    
    if self._use_torch:
        self.model = net  # PyTorch model
    else:
        self.model = self._fallback_model
    
    return self

def predict(self, X: pd.DataFrame) -> np.ndarray:
    if not isinstance(self.model, type(self)) and hasattr(self.model, "predict"):
        # Safe fallback
        return self.model.predict(X)
    # PyTorch path...
```

---

## 🟡 IMPORTANT IMPROVEMENTS

### 7. Unused Import `DNF_BASE_PROB_PER_RACE` in predict.py
**File**: [src/predict.py](src/predict.py#L15)  
**Issue**: Imported but only used in heuristic fallback, not passed to Monte Carlo.

```python
from config.settings import MODEL_DIR, PROC_DIR, MC_SIMULATIONS  # Missing DNF_BASE_PROB_PER_RACE
# Later...
preds = {
    "dnf_prob": np.full(len(feature_df), DNF_BASE_PROB_PER_RACE),  # Undefined
}
```

**Fix**: Either import it or define the default locally:
```python
from config.settings import (
    MODEL_DIR, PROC_DIR, MC_SIMULATIONS, DNF_BASE_PROB_PER_RACE
)
```

---

### 8. Strategy Assignment Too Simplistic
**File**: [src/predict.py](src/predict.py#L70)  
**Issue**: Driver strategy is chosen only by `deg_slope_medium`, ignoring quali-to-practice delta, fuel burn, and circuit profile.

```python
if deg_slope < 0.06:
    strategy = STRATEGY_OPTIONS["1-stop: M→H"]
elif deg_slope < 0.10:
    strategy = STRATEGY_OPTIONS["2-stop: S→M→H"]
else:
    strategy = STRATEGY_OPTIONS["2-stop: S→H→M"]
```

**Why it matters**: Non-optimal strategies reduce prediction accuracy and don't reflect real team decisions.

**How to fix**:
```python
def select_strategy(
    feature_row: pd.Series,
    circuit: CircuitProfile,
) -> list[tuple[str, int]]:
    """Multi-factor strategy selection."""
    deg = feature_row.get("deg_slope_medium", 0.08)
    qual_practice_delta = feature_row.get("qual_practice_delta", 0.0)
    overtaking = circuit.overtaking_factor
    
    # Low deg + good overtaking = aggressive 1-stop
    if deg < 0.05 and qual_practice_delta < 0.2 and overtaking > 1.2:
        return STRATEGY_OPTIONS["1-stop: S→H"]
    
    # Medium deg = standard 2-stop
    elif deg < 0.10:
        # Check rain probability for wet tire consideration
        if circuit.weather_rain_prob > 0.20:
            return STRATEGY_OPTIONS["2-stop: S→M→H"]
        else:
            return STRATEGY_OPTIONS["2-stop: S→M→H"]
    
    # High deg = conservative multi-stop
    else:
        return STRATEGY_OPTIONS["2-stop: S→H→M"]
```

---

### 9. FastF1 Loader Doesn't Handle Sprint Weeks
**File**: [src/ingestion/fastf1_loader.py](src/ingestion/fastf1_loader.py#L37)  
**Issue**: `load_weekend()` tries to load `FP3` but modern F1 has sprint weeks where FP3 is replaced by sprint.

```python
def load_weekend(year: int, gp: str | int) -> dict[str, fastf1.core.Session]:
    sessions: dict[str, fastf1.core.Session] = {}
    for label in ["FP1", "FP2", "FP3", "Q", "R"]:  # No sprint handling
        try:
            sessions[label] = load_session(year, gp, label)
        except Exception as exc:
            log.warning("Session %s not available: %s", label, exc)
    return sessions
```

**Fix**:
```python
def load_weekend(year: int, gp: str | int) -> dict[str, fastf1.core.Session]:
    sessions: dict[str, fastf1.core.Session] = {}
    
    # Try normal schedule first
    main_sessions = ["FP1", "FP2", "FP3", "Q", "R"]
    
    # Check if this is a sprint week (2021 onwards, usually ~6 weekends per season)
    # Try loading sprint qualifying + race
    sprint_sessions = ["FP1", "Q", "SQ", "R", "S"]  # Sprint week: Q, SQ, then race or sprint race
    
    for label in main_sessions:
        try:
            sessions[label] = load_session(year, gp, label)
        except Exception as exc:
            log.debug("Session %s not available: %s", label, exc)
    
    # If FP3 not available, try sprint week format
    if "FP3" not in sessions:
        for label in ["SQ", "S"]:
            try:
                sessions[label] = load_session(year, gp, label)
            except Exception:
                pass
    
    log.info("Loaded sessions: %s", list(sessions.keys()))
    return sessions
```

---

### 10. No Connection Pooling/Reuse in OpenF1 Client
**File**: [src/ingestion/openf1_client.py](src/ingestion/openf1_client.py#L37)  
**Issue**: Creates new `AsyncClient` on each call, losing connection pooling benefits.

```python
async def get(self, endpoint: str, **params: Any) -> list[dict]:
    async with httpx.AsyncClient(timeout=self.timeout) as client:
        # New connection on every call
        resp = await client.get(url)
```

**Fix**:
```python
class OpenF1Client:
    def __init__(self, base_url: str = OPENF1_BASE_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared async client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
    
    async def get(self, endpoint: str, **params: Any) -> list[dict]:
        client = await self._get_client()
        # ... use client ...
```

---

### 11. Circuit Profile Hardcoding Should Use Database
**File**: [src/predict.py](src/predict.py#L43)  
**Issue**: Circuit profiles are hardcoded in a dict. Should be data-driven.

```python
CIRCUIT_PROFILES: dict[str, CircuitProfile] = {
    "bahrain": CircuitProfile("Bahrain International Circuit", 57, ...),
    # ... 8 more circuits hardcoded ...
}
```

**Why it matters**: Adding new circuits or tweaking safety car rates requires code changes.

**Fix**:
```python
# Create config/circuits.json
{
  "bahrain": {
    "circuit_name": "Bahrain International Circuit",
    "total_laps": 57,
    "safety_car_rate": 0.05,
    "overtaking_factor": 1.2,
    "weather_rain_prob": 0.02
  }
}

# In predict.py
def load_circuit_profiles(config_path: Optional[Path] = None) -> dict[str, CircuitProfile]:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "circuits.json"
    
    if not config_path.exists():
        log.warning("No circuit config found — using defaults")
        return _default_profiles()
    
    with open(config_path) as f:
        data = json.load(f)
    
    profiles = {}
    for key, cfg in data.items():
        profiles[key] = CircuitProfile(
            name=cfg["circuit_name"],
            total_laps=cfg["total_laps"],
            safety_car_rate=cfg.get("safety_car_rate", 0.06),
            overtaking_factor=cfg.get("overtaking_factor", 1.0),
            weather_rain_prob=cfg.get("weather_rain_prob", 0.0),
        )
    return profiles

CIRCUIT_PROFILES = load_circuit_profiles()
```

---

### 12. No Logging Centralization
**File**: Multiple  
**Issue**: Each module sets up logging independently. No unified format/level control.

```python
# In fastf1_loader.py
log = logging.getLogger(__name__)

# In train.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)

# In predict.py (different format!)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
```

**Fix**: Create a central logging config:
```python
# src/logging_config.py
import logging

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure unified logging for all modules."""
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)

# In train.py, predict.py, etc.
from src.logging_config import setup_logging
setup_logging(level="INFO", log_file=PROC_DIR / "f1_predictor.log")
```

---

### 13. No Configuration Validation at Startup
**File**: [config/settings.py](config/settings.py#L1)  
**Issue**: `settings.py` doesn't validate that required paths/directories can be created or that dependencies are importable.

```python
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
# ... etc ...

# No checks for:
# - Writeable directories
# - Importable dependencies (torch, xgboost, etc.)
# - Valid configuration values
```

**Fix**:
```python
def validate_config() -> None:
    """Run at module import to catch config issues early."""
    errors = []
    
    # Check writeable directories
    for name, path in [
        ("DATA_DIR", DATA_DIR),
        ("MODEL_DIR", MODEL_DIR),
        ("CACHE_DIR", CACHE_DIR),
    ]:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            errors.append(f"{name} not writable: {e}")
    
    # Check key dependencies
    try:
        import fastf1  # noqa
    except ImportError:
        errors.append("fastf1 not installed")
    
    try:
        import torch  # noqa
    except ImportError:
        import warnings
        warnings.warn("PyTorch not installed — will use LightGBM fallback for NN models")
    
    # Validate parameter ranges
    if MC_RANDOM_SEED < 0:
        errors.append("MC_RANDOM_SEED must be >= 0")
    
    if errors:
        raise RuntimeError(f"Configuration errors:\n  " + "\n  ".join(errors))

# At module level
validate_config()
```

---

## 🟠 CODE QUALITY ISSUES

### 14. Magic Numbers Scattered Throughout
**Files**: Multiple (especially [src/simulation/monte_carlo.py](src/simulation/monte_carlo.py#L60))  
**Issue**: Hardcoded constants like pit times, lap limits, and pace adjustments.

```python
COMPOUND_OFFSETS = {
    "SOFT":         0.0,
    "MEDIUM":       0.6,  # Why 0.6s?
    "HARD":         1.1,
    "INTERMEDIATE": 2.0,
    "WET":          3.5,
}
```

**Fix**: Document or centralize in a `SIMULATION_PARAMS` dict in settings:
```python
# config/settings.py
SIMULATION_PARAMS = {
    # Tire compound pace offsets (baseline SOFT = 0.0)
    "compound_pace_offsets": {
        "SOFT":         0.0,     # reference compound
        "MEDIUM":       0.6,     # ~1.5 sec/km pace loss
        "HARD":         1.1,
        "INTERMEDIATE": 2.0,     # mid-range pace in wet
        "WET":          3.5,     # full rain pace
    },
    "compound_deg_multipliers": {  # relative to base driver deg
        "SOFT":         2.0,     # degrades quickly
        "MEDIUM":       1.0,     # reference degradation
        "HARD":         0.5,     # lasts longer
        "INTERMEDIATE": 0.8,
        "WET":          0.6,     # minimal deg in water
    },
    "pit_time_bounds": {
        "min_sec": 18.0,         # physical minimum
        "max_sec": 35.0,         # safety upper bound
    },
    "safety_car_duration_laps": (3, 5),  # SC period length
    "fuel_correction_factor": 0.08,      # sec/lap per 1.5 kg fuel
}
```

---

### 15. Inconsistent Type Hints
**Files**: [src/ingestion/](src/ingestion/)  
**Issue**: Some functions missing return type hints or using `str | int` instead of `Union[str, int]` (Python 3.10+ syntax used inconsistently).

```python
def load_session(year: int, gp: str | int, session: str) -> fastf1.core.Session:
    # Good: uses modern syntax

def extract_qualifying_times(session: fastf1.core.Session) -> pd.DataFrame:
    # Good

def build_elo_ratings(historical_results: pd.DataFrame) -> tuple[EloRating, EloRating]:
    # But if Python < 3.10, `tuple[...]` syntax breaks
```

**Fix**: Either enforce Python 3.10+ in pyproject.toml or use `from __future__ import annotations` at top of all files (already done, so this is OK).

---

### 16. Excessive Use of `fillna()` Without Explanation
**File**: [src/features/engineer.py](src/features/engineer.py#L200)  
**Issue**: Missing values are silently filled with defaults, hiding data quality issues.

```python
def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, default in IMPUTATION_DEFAULTS.items():
        if col not in df.columns:
            continue
        if default is None:
            median = df[col].median()
            if pd.isna(median):
                median = 0.0
            df[col] = df[col].fillna(median + 0.3)  # Why +0.3?
        else:
            df[col] = df[col].fillna(default)
    return df
```

**Why it matters**: Silent imputation masks data quality issues and can introduce bias.

**Fix**:
```python
def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with domain defaults. Logs imputation stats."""
    df = df.copy()
    imputation_log = {}
    
    for col, default in IMPUTATION_DEFAULTS.items():
        if col not in df.columns:
            continue
        
        num_missing = df[col].isna().sum()
        if num_missing == 0:
            continue
        
        if default is None:
            # Use median + penalty to flag imputed values
            median = df[col].median()
            if pd.isna(median):
                median = 0.0
                log.warning("Column %s is entirely null — using 0.0", col)
            penalty = 0.3  # Document: penalty for missing data
            df[col] = df[col].fillna(median + penalty)
            imputation_log[col] = (num_missing, f"median+{penalty}")
        else:
            df[col] = df[col].fillna(default)
            imputation_log[col] = (num_missing, default)
    
    if imputation_log:
        log.info("Imputed missing values:")
        for col, (count, strategy) in sorted(imputation_log.items()):
            log.info("  %s: %d missing → %s", col, count, strategy)
    
    return df
```

---

### 17. No Docstring for Complex Functions
**File**: [src/simulation/monte_carlo.py](src/simulation/monte_carlo.py#L130)  
**Issue**: `simulate_race()` is complex but lacks detailed docstring explaining lap-by-lap simulation logic.

```python
def simulate_race(
    drivers: list[DriverProfile],
    circuit: CircuitProfile,
    rng: np.random.Generator,
    rain: bool = False,
) -> pd.DataFrame:
    """Simulate one complete race. Returns a DataFrame with per-driver outcomes."""
    # No explanation of:
    # - How pit stops are modeled
    # - How safety cars affect lap times
    # - How ties are broken
    # - Lap-time noise distribution
```

**Fix**:
```python
def simulate_race(
    drivers: list[DriverProfile],
    circuit: CircuitProfile,
    rng: np.random.Generator,
    rain: bool = False,
) -> pd.DataFrame:
    """
    Simulate one complete race with stochastic lap times, pit stops, and incidents.
    
    Simulation process:
    1. For each driver, sample a DNF event (probability from dnf_prob)
    2. Build stint schedule from driver.strategy, extending last stint to total_laps
    3. Pre-generate safety car periods (3-5 lap bunches, ~circuit.safety_car_rate per lap)
    4. For each lap in each stint:
       - Compute base lap time from:
         * driver.base_pace_s + compound_offset + degradation_slope (stint_lap)
         * + noise (N(0, driver.pace_std))
         * + wet_skill adjustment if rain/SC period
       - If safety car lap: override to ~80s pace
       - Accumulate time, lap count
    5. Add pit stop time (N(team_pit_time, team_pit_std), min 18s)
    6. Sort by laps completed (classified > DNF), then total time
    7. Assign positions 1-20+ with official finishing order rules
    
    Returns:
        DataFrame with columns:
        - driver_code: driver identifier
        - total_time_s: total race time (or sentinel 1e9 for DNF)
        - dnf: boolean mechanical failure
        - laps_completed: number of completed laps
        - pit_stops: number of pit stops
        - finish_position: official position (1-20+)
    
    Args:
        drivers: list of DriverProfile objects with pace, strategy, dnf_prob
        circuit: CircuitProfile with total_laps, safety_car_rate
        rng: numpy random number generator for reproducibility
        rain: whether to apply rain weather modifiers
    """
```

---

### 18. Tuple Unpacking Without Index Labels
**File**: [src/features/engineer.py](src/features/engineer.py#L350)  
**Issue**: Functions return tuples but don't use named tuples, making unpacking error-prone.

```python
def build_feature_matrix(...) -> tuple[pd.DataFrame, dict]:
    # Caller must remember order:
    X, artifacts = build_feature_matrix(...)  # What if swapped?
```

**Fix**:
```python
from typing import NamedTuple

class FeatureMatrixResult(NamedTuple):
    X: pd.DataFrame
    artifacts: dict[str, any]

def build_feature_matrix(...) -> FeatureMatrixResult:
    # ... implementation ...
    return FeatureMatrixResult(X=df, artifacts=artifacts)

# Caller
result = build_feature_matrix(...)
X = result.X
artifacts = result.artifacts
```

---

## 📚 DOCUMENTATION ISSUES

### 19. README Missing Setup Prerequisites
**File**: [README.md](README.md#L60)  
**Issue**: Quick start doesn't mention Python version or system dependencies.

```markdown
## Quick start
# 1. Install dependencies
pip install -r requirements.txt
```

**Missing information**: Python 3.10+ requirement, fastf1 credentials, system libraries (GDAL for some geo features if used), etc.

**Fix**:
```markdown
## Requirements
- **Python**: 3.10 or later
- **System packages**: 
  - Linux/Mac: `libffi-dev` (for some dependencies)
  - Windows: Visual C++ Build Tools (for some packages)

## Setup
1. Clone repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Verify FastF1 cache directory exists:
   ```bash
   mkdir -p data/cache/fastf1
   ```
5. Test imports:
   ```bash
   python -c "import fastf1; import torch; print('✓ Dependencies OK')"
   ```
```

---

### 20. No API Documentation for Key Classes
**File**: [src/models/ensemble.py](src/models/ensemble.py#L250)  
**Issue**: `F1StackingEnsemble` class is complex but has no full docstring example of training + inference workflow.

**Fix**:
```python
class F1StackingEnsemble:
    """
    5-fold stacking ensemble for F1 race outcome prediction.
    
    Architecture:
        - Base models: XGBoost (position), LightGBM (pace), NN (tire deg)
        - Meta-learner: Ridge regression on OOF predictions
    
    Example usage:
        # Training
        ensemble = F1StackingEnsemble(n_splits=5)
        ensemble.fit(X_train, y_train, y_dnf=y_dnf)
        ensemble.save(Path("models/ensemble.pkl"))
        
        # Inference
        ensemble = F1StackingEnsemble.load(Path("models/ensemble.pkl"))
        preds = ensemble.predict(X_test)  # dict with keys: position_pred, dnf_prob, etc.
        
        # Feature importance
        fi = ensemble.feature_importance()
        print(fi.head(10))
    
    Attributes:
        base_models: list of base learners (PositionXGB, PaceLGBM, TireDegNN)
        incident_model: IncidentLogit for DNF/SC probability
        meta_learner: Ridge regressor combining base predictions
        _fitted: boolean, True if training complete
    """
```

---

## 🏗️ ARCHITECTURE/DESIGN ISSUES

### 21. Tight Coupling Between Feature Engineering and Config
**Files**: [src/features/engineer.py](src/features/engineer.py#L320) + [config/settings.py](config/settings.py)  
**Issue**: Feature engineering imports and directly uses `ALL_FEATURES` from settings, making it hard to test or extend.

```python
# In build_feature_matrix()
from config.settings import ALL_FEATURES
numeric_features = [c for c in ALL_FEATURES if c in df.columns]
```

**Problem**: ALL_FEATURES is a global that's easy to accidentally modify, and testing different feature sets is awkward.

**Fix**:
```python
class FeatureConfig:
    """Encapsulate feature list and defaults."""
    def __init__(self, feature_cols: list[str], imputation_defaults: dict):
        self.feature_cols = feature_cols
        self.imputation_defaults = imputation_defaults
    
    def filter_present(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.feature_cols if c in df.columns]

def build_feature_matrix(
    raw_df: pd.DataFrame,
    historical_results: Optional[pd.DataFrame] = None,
    fit: bool = True,
    artifacts: Optional[dict] = None,
    feature_config: Optional[FeatureConfig] = None,  # Inject config
) -> FeatureMatrixResult:
    feature_config = feature_config or _default_feature_config()
    # Use feature_config.feature_cols instead of ALL_FEATURES

# Default config from settings
def _default_feature_config() -> FeatureConfig:
    from config.settings import ALL_FEATURES, IMPUTATION_DEFAULTS
    return FeatureConfig(ALL_FEATURES, IMPUTATION_DEFAULTS)
```

---

### 22. Pickle-Based Model Serialization Is Fragile
**File**: [src/models/ensemble.py](src/models/ensemble.py#L380)  
**Issue**: Models are pickled directly, which is:
- Unsafe (pickle can execute code)
- Version-sensitive (breaks if library versions change)
- Non-portable (PyTorch models between systems)

```python
def save(self, path: Optional[Path] = None) -> Path:
    path = path or MODEL_DIR / "ensemble.pkl"
    with open(path, "wb") as f:
        pickle.dump(self, f)  # ❌ Unsafe
    return path
```

**Fix**: Use JSON + ONNX for safer serialization:
```python
def save(self, path: Optional[Path] = None) -> Path:
    """Save ensemble in a robust format."""
    path = path or MODEL_DIR / "ensemble"
    path.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "model_type": "F1StackingEnsemble",
        "base_models": [m.name for m in self.base_models],
        "n_splits": self.n_splits,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    
    # Save individual models in pickle (safer than pickled ensemble)
    for i, model in enumerate(self.base_models):
        with open(path / f"base_model_{i}.pkl", "wb") as f:
            pickle.dump(model, f)
    
    # Save meta-learner
    with open(path / "meta_learner.pkl", "wb") as f:
        pickle.dump(self.meta_learner, f)
    
    log.info("Ensemble saved to %s (metadata + components)", path)
    return path

@classmethod
def load(cls, path: Optional[Path] = None) -> "F1StackingEnsemble":
    path = path or MODEL_DIR / "ensemble"
    if not (path / "metadata.json").exists():
        # Fallback for old pickle format
        return cls._load_legacy_pickle(path.with_suffix('.pkl'))
    
    # Load metadata
    metadata = json.loads((path / "metadata.json").read_text())
    log.info("Loading ensemble v%s", metadata.get("version"))
    
    # Load components
    ensemble = cls(n_splits=metadata.get("n_splits", 5))
    # ...load models...
    return ensemble
```

---

### 23. No Version Control for Model Artifacts
**File**: [src/models/ensemble.py](src/models/ensemble.py#L380)  
**Issue**: Models are saved with fixed filenames, no way to track versions or rollback.

**Fix**:
```python
def save_versioned(self, base_path: Path | None = None) -> Path:
    """Save model with timestamp version."""
    base_path = base_path or MODEL_DIR
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version_dir = base_path / f"v{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    self._save_components(version_dir)
    
    # Create symlink to 'latest'
    latest_link = base_path / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(version_dir)
    
    log.info("Model saved to %s (linked as 'latest')", version_dir)
    return version_dir

# Usage
ensemble.save_versioned()  # v20260324_143022/
# Inference automatically uses 'latest' symlink
```

---

### 24. Global Mutable State in EloRating
**File**: [src/features/engineer.py](src/features/engineer.py#L30)  
**Issue**: `EloRating.ratings` is a mutable dict that persists across calls if not careful.

```python
class EloRating:
    def __init__(self):
        self.ratings: dict[str, float] = {}  # Shared reference risk
    
    def update_from_race(self, results_df: pd.DataFrame, entity_col: str = ...):
        # Modifies self.ratings in place
```

**Problem**: If multiple training runs share an EloRating object, they interfere.

**Fix**:
```python
class EloRating:
    def __init__(self, initial_ratings: Optional[dict] = None):
        self.ratings = dict(initial_ratings or {})  # Explicit copy
    
    def copy(self) -> "EloRating":
        """Create an independent copy."""
        return EloRating(self.ratings)
    
    # Usage:
    historical_elo = build_elo_ratings(historical_data)
    training_elo = historical_elo.copy()  # Don't modify original
```

---

## ⚡ PERFORMANCE ISSUES

### 25. No Batch Processing in Monte Carlo
**File**: [src/simulation/monte_carlo.py](src/simulation/monte_carlo.py#L230)  
**Issue**: Monte Carlo runs are sequential; could benefit from parallelization.

```python
iterator = tqdm(range(n_simulations), desc="Simulating races")
for _ in iterator:
    rain = rng.random() < rain_prob
    race_df = simulate_race(drivers, circuit, rng, rain=rain)  # Sequential
```

**Why it matters**: 10,000 simulations on 20 drivers is ~200k lap simulations, should be parallelizable.

**Fix**:
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def simulate_race_batch(
    args: tuple[list[DriverProfile], CircuitProfile, int, float],
) -> dict[str, list[int]]:
    """Single batch of simulations (picklable for multiprocessing)."""
    drivers, circuit, seed, rain_prob = args
    rng = np.random.default_rng(seed)
    
    finish_positions = {d.code: [] for d in drivers}
    dnf_counts = {d.code: 0 for d in drivers}
    
    # Run N simulations
    for _ in range(100):  # Batch size
        rain = rng.random() < rain_prob
        race_df = simulate_race(drivers, circuit, rng, rain=rain)
        for _, row in race_df.iterrows():
            finish_positions[row["driver_code"]].append(row["finish_position"])
            if row["dnf"]:
                dnf_counts[row["driver_code"]] += 1
    
    return finish_positions, dnf_counts

def run_monte_carlo_parallel(
    drivers: list[DriverProfile],
    circuit: CircuitProfile,
    n_simulations: int = MC_SIMULATIONS,
    seed: int = MC_RANDOM_SEED,
    n_workers: int = None,
) -> MonteCarloResults:
    """Run Monte Carlo with multiprocessing."""
    n_workers = n_workers or mp.cpu_count()
    batch_size = max(100, n_simulations // n_workers)
    n_batches = (n_simulations + batch_size - 1) // batch_size
    
    # Prepare batch arguments
    batch_args = [
        (drivers, circuit, seed + i, circuit.weather_rain_prob)
        for i in range(n_batches)
    ]
    
    # Run batches in parallel
    finish_positions = {d.code: [] for d in drivers}
    dnf_counts = {d.code: 0 for d in drivers}
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for pos_batch, dnf_batch in tqdm(
            executor.map(simulate_race_batch, batch_args),
            total=n_batches,
            desc="Monte Carlo (parallel)",
        ):
            for drv, positions in pos_batch.items():
                finish_positions[drv].extend(positions)
            for drv, count in dnf_batch.items():
                dnf_counts[drv] += count
    
    return MonteCarloResults(
        driver_codes=[d.code for d in drivers],
        finish_positions=finish_positions,
        dnf_counts=dnf_counts,
        n_simulations=n_simulations,
    )
```

---

### 26. Inefficient DataFrame Operations in Historical Scraper
**File**: [src/scraper/historical_scraper.py](src/scraper/historical_scraper.py#L150)  
**Issue**: Appends rows to DataFrame in loops, which is O(n²).

```python
records = []
for driver in laps["Driver"].unique():
    # ... compute values ...
    records.append(row)

df = pd.DataFrame(records)  # OK—this way is good
```

Actually, looking closer, they use list append then DataFrame constructor, which is fine. ✓

---

### 27. No Caching of Expensive Computations
**File**: [src/features/engineer.py](src/features/engineer.py#L60)  
**Issue**: ELO and circuit affinity recalculated on every call, even with same historical data.

**Fix**: Cache using `functools.lru_cache` or disk-based cache:
```python
from functools import lru_cache
import pickle

_ELO_CACHE = MODEL_DIR / "elo_cache"

@lru_cache(maxsize=2)
def build_elo_ratings_cached(
    historical_pickle: bytes,  # Serialize DF to bytes for cache key
) -> tuple[EloRating, EloRating]:
    """Build ELO with LRU cache to avoid recomputation."""
    df = pickle.loads(historical_pickle)
    driver_elo = EloRating()
    team_elo = EloRating()
    # ... computation ...
    return driver_elo, team_elo

# Usage
hist_bytes = pickle.dumps(historical_results)
driver_elo, team_elo = build_elo_ratings_cached(hist_bytes)
```

---

## 📋 SUMMARY TABLE

| Category | Issue Count | Severity | Estimated Effort |
|----------|------------|----------|-----------------|
| Critical | 6 | 🔴 | High (2-3 days) |
| Important | 6 | 🟡 | Medium (1-2 days) |
| Code Quality | 6 | 🟠 | Low (0.5-1 day) |
| Documentation | 2 | 📚 | Low (2-3 hours) |
| Architecture | 4 | 🏗️ | Medium (1-2 days) |
| Performance | 3 | ⚡ | Low-Medium (0.5-1 day) |

---

## 🎯 RECOMMENDED PRIORITY ORDER

### Phase 1 (Day 1 — Critical):
1. ✅ Add error handling to OpenF1Client (Issue #1)
2. ✅ Add input validation to feature engineering (Issues #2, #5)
3. ✅ Fix model artifact persistence (Issue #3)
4. ✅ Prevent data leakage in ELO (Issue #4)

### Phase 2 (Day 2 — Important):
5. ✅ Fix PyTorch model fallback (Issue #6)
6. ✅ Add logging centralization (Issue #12)
7. ✅ Support sprint weeks in FastF1 loader (Issue #9)
8. ✅ Improve strategy selection (Issue #8)

### Phase 3 (Day 3 — Nice-to-Have):
9. ✅ Switch to safer model serialization (Issue #22)
10. ✅ Add multiprocessing to Monte Carlo (Issue #25)
11. ✅ Centralize configuration validation (Issue #13)
12. ✅ Fix documentation gaps (Issues #19-20)

---

## 🔧 Quick Wins (< 1 hour each)

- [ ] Fix missing import: `DNF_BASE_PROB_PER_RACE` in `predict.py`
- [ ] Document magic numbers in `COMPOUND_OFFSETS`
- [ ] Add circuit profiles database (JSON instead of hardcoding)
- [ ] Improve docstrings for `simulate_race()` and `F1StackingEnsemble`

---

## ✅ POSITIVE NOTES

The project has several strengths worth highlighting:

1. **Good separation of concerns**: Ingestion, feature engineering, modeling, and simulation are nicely modularized.
2. **Comprehensive docstrings in README**: Architecture diagram and examples are clear and helpful.
3. **Defensive programming in many places**: Fallbacks for missing sessions, graceful degradation (e.g., FastF1 → OpenF1 → defaults).
4. **Type hints throughout**: Modern Python with `from __future__ import annotations`.
5. **Monte Carlo simulation is well-structured**: Clear DriverProfile/CircuitProfile data classes, reproducible with seeds.
6. **Robust scraper**: Rate limiting, retry logic, and progress tracking in historical_scraper.py are production-quality.

