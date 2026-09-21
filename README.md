# FAHRAI policy calibration

Code and synthetic data for *A method for recalibrating response probabilities under changes in task selection in adaptive learning systems*.

The experiment changes how tasks are selected while holding learner states and response probabilities fixed. It compares six estimators on a population of 12 state–action cells. Each of three logging policies uses 200 calibration samples of 6,000 observations. Evaluation sums over the known target population; it does not use a sampled test set.

The original calibration experiment is entirely synthetic. A separate LLM extension uses real model responses to synthetic exercises and replays call policies using the saved calibration predictions. Neither experiment uses learner records or measures learning over time.

## Run

The reference calculations used Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0 on Linux x86_64. Figures were checked with Matplotlib 3.10.8. `environment.json` records the verification environment; `requirements-lock.txt` pins the numerical and plotting dependencies.

From the repository root, using Python 3.12.14:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python -m fahrai_calibration verify
```

To regenerate every calibration sample and compare all predictions and numerical results with the saved evidence:

```sh
python -m fahrai_calibration verify --reproduce
```

To write a fresh set of results, CSV exports, and figures:

```sh
python -m fahrai_calibration run --output outputs
```

The output directory must be empty. The verification commands leave the reference files unchanged. An optional report can be saved with `--report outputs/verification.json`.

## Files

| Path | Contents |
| --- | --- |
| `src/fahrai_calibration/model.py` | Population, sampling, and constrained logistic fitting |
| `src/fahrai_calibration/experiment.py` | Six estimators, replication summaries, paired intervals, and weight clipping |
| `src/fahrai_calibration/metrics.py` | Exact population calibration and decision metrics |
| `src/fahrai_calibration/controls.py` | Profile-likelihood solver, bin and threshold sensitivity, and population controls |
| `src/fahrai_calibration/verification.py` | Transport, Brier, and decision identities; checks of saved results |
| `src/fahrai_calibration/figures.py` | The two manuscript figures |
| `data/results.json` | Summary statistics, paired intervals, and population clipping results |
| `data/replications.json` | 3,600 records, each containing one estimator's 12 predictions and metrics |
| `data/controls.json` | Saved sensitivity results and alternative-solver checks |
| `data/calibration_counts.csv` | 7,200 cell-count records across the 600 calibration samples |
| `data/predictions.csv` | 43,200 scalar predictions in long format |
| `data/population.csv` | Cell definitions, true probabilities, raw scores, and target masses |
| `data/table_1.csv`, `data/table_2.csv` | Numerical values underlying the two manuscript tables |
| `docs/data_dictionary.md` | Field definitions and sample construction |
| `docs/equations.tex` | Eight displayed equations; numbering differs between journal layouts |
| `figures/` | Publication figures at 300 dpi |
| `verification/reproduction.json` | The recorded verification run |
| `SHA256.json` | Checksums of the reference data and figures |

## What is checked

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

The LLM analysis uses the saved prediction vectors and passed independently on macOS arm64 and Linux x86_64. [Linux calibration verification](verification/calibration_linux.json), [macOS LLM verification](verification/llm.json), and [Linux LLM verification](verification/llm_linux.json) record these checks.

The original JSON evidence is retained unchanged. The `date` field in the original `controls.json` records the earlier audit date and is excluded from numerical comparisons. The alternative solver's residual is checked against the stated 2 × 10⁻⁸ bound; its precise value can change with floating-point evaluation. Execution details belong in the separate verification report.

## Scope of the implementation

The raw score omits the task action. Logistic corrections pool actions and use either one map or one map per evidence regime. The context-cell estimator uses action information and half-count smoothing. This is a designed comparison of information and coverage, not a ranking of deployed educational systems.

Randomness comes from NumPy's `default_rng` with PCG64. For each setting and seed, the generator first draws multinomial counts over the 12 cells, then binomial correctness counts conditional on those counts. These aggregated counts are sufficient for the implemented losses. Seeds are reused across logging settings; the six estimators within each sample share its counts.


## LLM execution extension

The extension contains 2,304 main responses: eight model configurations answer the same 144 synthetic exercises twice. It includes hosted models and Qwen 3.5 4B run locally on an M1 Pro. All received main answers are retained, including incorrect and invalid output.

```sh
python -m fahrai_calibration.llm verify
python -m fahrai_calibration.llm analyze --output outputs/llm
```

These commands make no network calls and require no API keys. Verification regenerates task labels with two independent solvers, regrades the final outputs, checks usage-based fee estimates, and compares regenerated analysis with the reference tables. The original calibration commands and evidence remain unchanged.

| Configuration | Successful responses | Success rate |
|---|---:|---:|
| GPT-4.1 mini | 185/288 | 64.24% |
| GPT-6 Astra | 288/288 | 100.00% |
| Claude Opus 5 | 288/288 | 100.00% |
| Gemini 3.1 Pro | 288/288 | 100.00% |
| GPT-5.6 Luna | 282/288 | 97.92% |
| Claude Sonnet 5 | 282/288 | 97.92% |
| Gemini 3.5 Flash-Lite | 283/288 | 98.26% |
| Qwen 3.5 4B, Q4_K_M, local | 281/288 | 97.57% |

Success requires the correct integer and the recorded JSON validity checks, including a nonempty explanation. Citation validity is checked separately. Explanation quality and the prompt's word limit are not part of this score. Observed perfect accuracy does not establish model equivalence. Configuration differences prevent attributing every difference to model identity alone.

The replay compares thresholds and equal call budgets. Learner-error mass remains simulated; LLM success, request latency and token usage are measured. Local API fees are zero, but total local compute cost was not measured.

- [Protocol, equations and limitations](docs/llm_experiment.md)
- [Data dictionary](docs/llm_data_dictionary.md)
- [Tasks, requests, final responses and configurations](data/llm/)
- [Reference analysis](data/llm/reference/)

The LLM module lives in `src/fahrai_calibration/llm/`. `data/llm/SHA256.json` checks the supplement independently of the original `SHA256.json`. The v1.0.0 release contains only the original mathematical experiment; cite the specific later commit that includes this extension when using the LLM results.
