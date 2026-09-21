# Reproduction

## Calibration checks

Verification reconstructs metrics from every saved prediction vector, checks the summary statistics and paired intervals, and regenerates the multinomial and binomial counts from seeds 0–199. A separate profile-likelihood solver checks 240 fitted vectors from 60 samples. Its largest prediction difference must remain below the manuscript's stated bound of 2 × 10⁻⁸.

The full reproduction command also refits every estimator in all 600 samples. Comparisons allow relative error 10⁻⁸ and absolute error 10⁻¹⁰ for floating-point arithmetic. The report separately records whether outputs were exactly equal. Other operating systems and numerical-library versions have not been validated; small numerical differences can affect predictions close to a bin boundary or decision threshold.

The saved calibration counts reproduce in the recorded Linux x86_64 environment. A macOS arm64 check with the same package versions produced different counts for some seeds. For the original calibration verification on a Mac, use the reference environment, for example:

```sh
docker run --rm --platform linux/amd64 \
  --mount type=bind,source="$PWD",target=/work,readonly \
  --workdir /work -e PYTHONPATH=/work/src \
  python:3.12.14-slim sh -c \
  'pip install -r requirements-lock.txt && python -m fahrai_calibration verify && python -m fahrai_calibration.llm verify'
```

The LLM analysis uses the saved prediction vectors and passed independently on macOS arm64 and Linux x86_64. [Linux calibration verification](../verification/calibration_linux.json), [macOS LLM verification](../verification/llm.json), and [Linux LLM verification](../verification/llm_linux.json) record these checks.

The original JSON evidence is retained unchanged. The `date` field in the original `controls.json` records the earlier audit date and is excluded from numerical comparisons. The alternative solver's residual is checked against the stated 2 × 10⁻⁸ bound; its precise value can change with floating-point evaluation. Execution details belong in the separate verification report.

## Scope of the implementation

The raw score omits the task action. Logistic corrections pool actions and use either one map or one map per evidence regime. The context-cell estimator uses action information and half-count smoothing. This is a designed comparison of information and coverage, not a ranking of deployed educational systems.

Randomness comes from NumPy's `default_rng` with PCG64. For each setting and seed, the generator first draws multinomial counts over the 12 cells, then binomial correctness counts conditional on those counts. These aggregated counts are sufficient for the implemented losses. Seeds are reused across logging settings; the six estimators within each sample share its counts.

## LLM checks

The LLM verification command checks 2,304 responses, regenerates 156 tasks with two independent solvers, validates token-based fee estimates, and compares both analysis samples with the reference CSVs. It runs offline on macOS arm64 and Linux x86_64. See the [protocol](llm_experiment.md) for scoring, uncertainty and interpretation.
