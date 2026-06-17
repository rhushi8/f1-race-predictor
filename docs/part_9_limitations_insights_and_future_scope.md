# Part 9: Limitations, Insights, and Future Scope

### 9.1 Current limitations
- Data drift across seasons due to regulation and team changes.
- Some engineered features are proxies, not direct latent variables.
- Simulation parameters include assumptions that may not fully match all circuits.
- API availability and schema shifts can create operational fragility.

### 9.2 Practical implementation challenges observed in projects like this
- temporal leakage from incorrectly scoped historical aggregates
- artifact incompatibility between training and inference versions
- noisy target behavior in chaotic race events
- balancing realism with computational efficiency

### 9.3 Key insights from this architecture
- Hybrid deterministic plus stochastic design is stronger than deterministic-only for race decision support.
- Feature engineering and temporal correctness often matter more than adding model complexity.
- Visualization quality and confidence framing are critical for user trust.

### 9.4 Reasonable future improvements
1. Add stronger temporal feature-store style controls and lineage metadata.
2. Add automatic drift monitoring by season segment.
3. Introduce richer weather and strategy-conditioned simulation scenarios.
4. Add ensemble uncertainty decomposition by feature groups.
5. Expand CI with leakage tests and dashboard contract tests.

### 9.5 How this section connects to whole project
Part 9 closes the learning loop by showing what to improve after initial deployment.
It makes the project interview-ready by demonstrating engineering maturity.

### 9.6 Part 9 key takeaways
- All predictive systems have assumptions and limits.
- Reliability and transparency are ongoing processes.
- Future scope should be prioritized by measurable impact.

### 9.7 Part 9 revision points
- Memorize at least three current limitations.
- Memorize at least three high-impact improvements.
- Be ready to discuss trade-offs of each improvement.

### 9.8 Part 9 interview and viva questions
1. What is the biggest current bottleneck in this project and why?
2. How would you prioritize the first two future improvements?
3. What is one limitation that cannot be fully solved with modeling alone?
4. How would you prove a future improvement is actually useful?
5. Which part of the pipeline is most failure-sensitive operationally?

### 9.9 Part 9 interview and viva answers
1. Temporal robustness across changing seasons is the biggest bottleneck because signal dynamics shift.
2. Prioritize leakage and drift controls first, then improve simulation realism tied to measurable gains.
3. External race randomness and incidents cannot be fully solved by deterministic modeling.
4. Validate with pre-registered metrics, walk-forward comparisons, and statistically meaningful deltas.
5. Ingestion and feature-contract boundaries are most failure-sensitive because all downstream stages depend on them.

---

