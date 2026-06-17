# Part 7: Monte Carlo Simulation and Prediction Logic

### 7.1 What this step is
This stage converts deterministic model outputs into probabilistic race-outcome distributions.
It simulates many possible race realizations under uncertainty.

### 7.2 Why it is important
Races are event-driven and stochastic.
A single predicted finishing order cannot represent safety cars, pit variance, weather, or reliability events.

### 7.3 How it works internally
1. Build DriverProfile for each driver:
- grid position
- base pace
- pace variance
- degradation slope
- DNF probability
- strategy template

2. For each simulation run:
- sample DNF events
- simulate lap times by compound, degradation, and noise
- apply pit stop time losses
- apply safety-car effects
- rank by completed laps and total time

3. Aggregate over many runs:
- win probability
- podium probability
- points probability
- expected finish and confidence interval

### 7.4 Tools and libraries used
- NumPy random generation and numeric ops
- Pandas result aggregation
- project simulation engine in src/simulation/monte_carlo.py

### 7.5 Simple dummy simulation example
If Driver A has deterministic predicted position 1.8 and Driver B has 2.3:
- deterministic ranking suggests A ahead.
- after 10,000 simulation runs with reliability and pit variability,
  A might win 41% and B might win 33%.

This reflects realistic uncertainty instead of false certainty.

### 7.6 Workflow summary
Model predictions -> Calibration -> Driver profiles -> Monte Carlo runs -> Probability summary.

### 7.7 Practical challenges and assumptions
- Assumption: chosen noise and event parameters approximate race behavior.
- Limitation: simulation fidelity depends on quality of upstream model and handcrafted parameters.
- Challenge: balancing runtime cost with simulation count.

### 7.8 How this connects to overall project
Part 7 is the decision-facing layer that transforms model outputs into interpretable probabilities for Part 8 dashboard.

### 7.9 Part 7 key takeaways
- Simulation is core output logic, not optional decoration.
- Probabilities and intervals communicate uncertainty better than single rank.
- Calibration before simulation improves plausibility.

### 7.10 Part 7 revision points
- Memorize DriverProfile inputs.
- Remember event types included in each simulation run.
- Remember the four primary aggregated output probabilities.

### 7.11 Part 7 interview and viva questions
1. Why not directly map model outputs to win probabilities without simulation?
2. How do DNF and pit variance influence distribution tails?
3. What is the minimum simulation count you would use and why?
4. How would you validate that simulation assumptions are realistic?
5. Where does calibration sit relative to simulation and why?

### 7.12 Part 7 interview and viva answers
1. Direct mapping misses event uncertainty and cannot represent race outcome distributions.
2. They increase downside risk and spread, creating heavier tails in finish outcomes.
3. Start with around 2,000 for development and move to 10,000 for stable production summaries.
4. Back-check simulated distributions against historical race outcome statistics and scenario tests.
5. Calibration is applied before simulation so sampled race dynamics start from realistic baseline ranks.

---

