# Part 3: Data Cleaning and Preprocessing

### 3.1 What this step is
Preprocessing converts raw merged data into a model-safe table.
It enforces data types, removes impossible values, handles missingness, and standardizes categories.

### 3.2 Why it is important
Gradient boosting models and simulation logic assume coherent numeric inputs.
Preprocessing errors produce unstable outputs and misleading probability distributions.

### 3.3 How it works internally
1. Basic validation:
- verify required columns such as driver_code, team_name, year, gp where needed
- ensure non-empty DataFrame

2. Type normalization:
- parse numeric fields safely
- convert categorical fields to consistent strings

3. Missing value handling:
- domain-specific defaults for features such as tire degradation and weather
- median-plus-penalty style imputation for qualifying segments where needed

4. Category encoding:
- label encode selected categorical columns
- reuse training encoders during inference

5. Scaling:
- fit scaler during training
- reuse scaler and expected feature names during inference

### 3.4 Tools and libraries used
- Pandas: value parsing, null handling, type alignment
- NumPy: numeric replacement and finite checks
- Scikit-learn: LabelEncoder and StandardScaler

### 3.5 Simple dummy example
Input row:
- driver_code: "ver "
- q3_gap_to_pole: null
- deg_slope_medium: inf

After preprocessing:
- driver_code: "VER"
- q3_gap_to_pole: imputed numeric value
- deg_slope_medium: converted from inf to null, then imputed

### 3.6 Workflow summary
Validate schema -> Clean values -> Impute -> Encode -> Scale -> Emit model-ready matrix.

### 3.7 Practical challenges and assumptions
- Assumption: defaults are representative enough for missing optional signals.
- Challenge: unknown categories at inference must be mapped safely.
- Challenge: avoid data leakage by fitting preprocessors only on training scope.

### 3.8 How this connects to overall project
Part 3 produces the exact feature matrix consumed by Part 5 model training and Part 7 simulation profile generation.

### 3.9 Part 3 key takeaways
- Preprocessing is part of model behavior, not a side utility.
- Train-time and inference-time preprocessing must remain aligned.
- Domain-aware defaults improve robustness.

### 3.10 Part 3 revision points
- Remember the order: validate, clean, impute, encode, scale.
- Remember that encoder/scaler persistence is mandatory.
- Remember that infinities must be normalized before imputation.

### 3.11 Part 3 interview and viva questions
1. Why is inference-time refitting of encoders a serious bug?
2. How do you design imputation defaults for motorsport data?
3. What is the risk of preprocessing with full-dataset statistics?
4. How do you handle unseen categorical labels at inference?
5. Why does preprocessing belong in artifacts with the model?

### 3.12 Part 3 interview and viva answers
1. Refitting changes category mappings and breaks model input consistency.
2. Use domain-informed defaults based on realistic race ranges and historical behavior.
3. It leaks future information and inflates apparent model quality.
4. Map unseen values to known fallback classes or safe default encodings.
5. Inference must reproduce the exact training transformations to remain valid.

---

