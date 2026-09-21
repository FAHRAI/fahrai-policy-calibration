# FAHRAI policy calibration

Code and data for response-probability recalibration under changes in task selection.

The calibration experiment compares six estimators across 12 state–action cells and three logging policies. The LLM experiment evaluates call policies using 2,304 recorded responses from eight configurations on 144 synthetic exercises. Both use synthetic tasks and learner probabilities.

## Setup

Requires Python 3.12. The reference calibration environment is Python 3.12.14 on Linux x86_64; dependencies are pinned in `requirements-lock.txt`.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

## Calibration experiment

```sh
python -m fahrai_calibration verify
python -m fahrai_calibration verify --reproduce
python -m fahrai_calibration run --output outputs/calibration
```

`verify` checks the saved results; `--reproduce` also refits all 600 calibration samples. `run` writes new results and figures to an empty output directory.

Reproducing the saved random counts requires the reference Linux x86_64 environment. See [reproduction](docs/reproduction.md) for a Docker command and platform details.

## LLM experiment

```sh
python -m fahrai_calibration.llm verify
python -m fahrai_calibration.llm analyze --output outputs/llm
```

These commands regrade the recorded answers and recompute the analysis offline. They require no API keys. The analysis output directory must not already exist.

- [Results](docs/llm_results.md)
- [Protocol](docs/llm_experiment.md)
- [Data dictionary](docs/llm_data_dictionary.md)

## Repository structure

| Path | Contents |
|---|---|
| `src/fahrai_calibration/` | Calibration models, metrics, controls and figures |
| `src/fahrai_calibration/llm/` | Task generation, grading, policy replay and verification |
| `data/` | Calibration samples, predictions and reference results |
| `data/llm/` | Model configurations, exact requests, final responses and reference tables |
| `docs/` | Equations, protocols and data definitions |
| `figures/` | Manuscript figures |
| `tests/` | Model, grading and replay tests |
| `verification/` | Recorded verification reports |

`SHA256.json` and `data/llm/SHA256.json` contain checksums for the two experiments.

## Tests

```sh
python -m unittest discover -s tests -v
```

## Citation

See [CITATION.cff](CITATION.cff) and cite the commit used for reproduction. Release v1.0.0 contains the calibration experiment only; the LLM results require a later commit.
