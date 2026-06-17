# Part 2: Data Sources and Acquisition Using FastF1 and OpenF1

### 2.1 What this step is
Data acquisition is the stage where race-weekend and historical signals are collected before feature engineering.
In this project, two main sources are combined:
- FastF1 for structured session-level and lap-level motorsport data
- OpenF1 for endpoint-based live and event enrichment

### 2.2 Why it is important
Model quality is limited by data quality.
If acquisition is incomplete, inconsistent, or unstable, downstream modeling accuracy and simulation realism drop quickly.
This stage also controls reproducibility through caching and schema normalization.

### 2.3 How it works internally
1. FastF1 session loading:
- Load weekend sessions such as FP1, FP2, FP3, Qualifying, Race.
- On sprint weekends, handle alternate sessions safely.

2. FastF1 extraction:
- qualifying times and gaps
- grid positions
- practice lap pace and long-run estimates
- tire degradation proxies
- weather statistics

3. OpenF1 enrichment:
- resolve race session key
- fetch drivers, stints, pit stops, positions, and weather endpoints
- compute per-team pit averages and per-driver strategy summaries

4. Merge and standardize:
- merge by stable keys such as driver code and team where appropriate
- normalize schema to project-standard column names
- add metadata such as year, race name, circuit id

### 2.4 Tools and libraries used
- FastF1: core race-weekend telemetry and timing source
- OpenF1 API (Application Programming Interface): enrichment endpoints
- httpx: network calls and timeout/error handling for OpenF1 client
- Pandas: merge, transformation, and schema standardization
- Python logging: ingestion-stage diagnostics

### 2.5 FastF1 vs OpenF1 usage in this project
FastF1 is used for structured session-derived features.
OpenF1 is used for enrichment and operational race context.

FastF1 strengths in this project:
- mature session abstraction
- rich lap-time and timing context
- direct qualifying and practice extraction

OpenF1 strengths in this project:
- endpoint flexibility
- live-friendly race event data
- pit stop and stint summaries

Combined design benefit:
- FastF1 provides stable baseline structure
- OpenF1 adds contextual depth where available

### 2.6 Simple dummy example
Dummy rows before merge:

FastF1 table:
- driver_code: VER, LEC
- q3_gap_to_pole: 0.000, 0.121
- fp2_long_run_pace_s: 91.20, 91.34

OpenF1 table:
- driver_code: VER, LEC
- first_compound: SOFT, MEDIUM
- pit_time_mean_s: 21.8, 22.4

Merged table output:
- driver_code, q3_gap_to_pole, fp2_long_run_pace_s, first_compound, pit_time_mean_s

This merged output becomes one input source for feature engineering.

### 2.7 Workflow summary
Acquire -> Validate -> Normalize -> Merge -> Enrich -> Emit canonical DataFrame.

### 2.8 Acquisition challenges and trade-offs
- API rate limits and timeout behavior
- session availability differences across weekends
- inconsistent driver/team naming across sources
- optional endpoint data that may be missing

Trade-off:
- strict hard-fail policy improves quality control but can reduce run continuity
- graceful degradation improves availability but requires clear warnings

### 2.9 How this connects to overall project
This stage feeds Part 3 preprocessing and Part 4 feature engineering.
If this stage is weak, no model or simulation strategy can recover lost signal quality.

### 2.10 Part 2 key takeaways
- FastF1 and OpenF1 are complementary in this architecture.
- Canonical schema and stable keys are essential for reliable merges.
- Acquisition reliability is an engineering task, not just a data task.

### 2.11 Part 2 revision points
- Remember what each source contributes.
- Remember why driver key normalization is mandatory.
- Remember the acquisition lifecycle from fetch to canonical output.

### 2.12 Part 2 interview and viva questions
1. Why use both FastF1 and OpenF1 instead of only one source?
2. What ingestion failures should be hard-fail versus soft-fail?
3. How do you keep schema stable when source providers evolve fields?
4. Why is caching important in race analytics pipelines?
5. How would you test ingestion reliability without race-day dependencies?

### 2.13 Part 2 interview and viva answers
1. FastF1 provides robust session structure while OpenF1 adds enrichment depth; together they improve coverage.
2. Missing required keys should hard-fail, while optional enrichment gaps should soft-fail with warnings.
3. Use canonical column mapping, schema validation, and fallback defaults for renamed or missing fields.
4. Caching reduces latency, API dependence, and run-to-run variability during iterative development.
5. Use fixed historical weekends, mocked API responses, and schema contract tests in automated checks.

---

