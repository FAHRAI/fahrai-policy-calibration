# LLM execution experiment

This extension evaluates the execution layer of a calibration-based call policy. The original mathematical experiment is unchanged. The public bundle contains synthetic exercises and real model outputs; it contains no learner records.

## Design

There are 12 development exercises and 144 main exercises. Each of the 12 state–action cells receives one exercise from each combination of four families and three structural difficulty levels. Each configuration answers every main exercise twice: 288 responses per configuration, 2,304 in total. Development exercises are excluded from the published main response sample and every reported comparison.

The families are exact arithmetic, sequential state updates, first-matching ordered rules, and value maximization subject to weight, cost and category constraints. Every gold answer is reproduced by two deterministic solvers. Task generation uses Python's `random.Random(20260922)`; assignment order uses `random.Random(20260923)`. Both are checked against the saved bundle.

The task-to-cell mapping is a synthetic engineering assignment. It is not evidence that actual learners in a particular state would receive these exercises or have the assigned success probability. No expert or LLM judge supplies the gold labels.

Only the exercise and reference rule are sent to a model. Gold answers, cell identifiers, difficulty labels and simulated learner probabilities are withheld. `requests.jsonl` stores the exact main request bodies; `prompt.json` stores the common instruction and response schema. Responses are independent chats without tools or prior conversation.

## Configurations

| Configuration | Model identifier | Output cap | Reasoning | Concurrency |
|---|---|---:|---|---:|
| baseline | `gpt-4.1-mini-2025-04-14` | 600 | Not requested | 1 |
| astra | `gpt-6-astra` | 16,384 | High | 4 |
| opus | `claude-opus-5` | 16,384 | Adaptive, high effort | 4 |
| pro | `gemini-3.1-pro-preview` | 16,384 | High | 4 |
| luna | `gpt-5.6-luna` | 16,384 | High | 4 |
| sonnet | `claude-sonnet-5` | 16,384 | Adaptive, high effort | 4 |
| lite | `gemini-3.5-flash-lite` | 16,384 | High | 4 |
| qwen_local | `qwen3.5:4b` | 16,384 | Thinking enabled | 1 |

Concurrency is the configured maximum per provider. Continuations and service scheduling can change actual overlap. The initial strong-model development pilot used an 8,192-token cap; this was increased before its main collection. Smaller hosted models used a separate 12-exercise development pilot. The local extension was added after the hosted results were observed. These are exploratory extensions, not an independently preregistered model ranking.

Qwen uses Ollama 0.34.2, Q4_K_M weights, a 24,576-token context, and an Apple M1 Pro with 16 GiB unified memory on macOS 14.5. The full weight digest and saved sampling defaults are in `configurations.json`. Model metadata reports 4.7B parameters for the model marketed as 4B. The quantized model, runtime, hardware and settings jointly define this configuration.

Provider-specific reasoning controls are not equivalent experimental treatments. Hosted model aliases can change; the bundle records requested and returned identifiers and collection timestamps. Repeating the requests need not produce identical answers or timings.

## Scoring and measurements

`answer_exact` requires a completed response with a JSON object containing exactly `final_answer`, `explanation` and `source_ids`, an integer answer, a nonempty explanation, and a list of string source identifiers. The integer must match the deterministic gold answer. This reproduces the collection-time grading rule.

Correct citation identifiers are checked separately in `source_ids_valid`. The prompt requests an explanation of at most 100 words, but the primary grader does not enforce word count or assess explanation quality, pedagogy, or consistency with the number. Consequently, the success rate is not a measure of fully correct teaching explanations. Truncated responses are failures even if a JSON object could be recovered. Received wrong or invalid answers are retained; there is no outcome-based retry or response selection. Transport failures are not model answers; the published sample contains one received response per scheduled assignment, with no imputation.

API fee estimates use recorded input/output usage and the historical rates in `configurations.json`. Cached inputs are priced at the full input rate; cache-write tokens receive the recorded 25% input premium. These are usage-based estimates, not invoice totals or current price quotations. Output usage includes reasoning tokens where the provider accounts for them. Tokenizers and usage conventions differ between providers.

The local API fee is zero. Its total compute cost is unknown, not zero: electricity and hardware costs were not measured. Local resource fields include model loading and generation durations, generation throughput and allocation reported by Ollama. This allocation is not whole-system peak memory. Prefix caching may occur. End-to-end latency includes loading where applicable and is specific to the recorded execution conditions.

## Calibration-policy replay

For cell `c`, let `w[c]` be target mass, `p[c]` simulated learner success probability, `r[c]` call probability, and `e[c]`, `d[c]`, `t[c]` the measured cell means for LLM success, API fee and request latency. The replay computes:

- Call rate: `C = sum(w * r)`.
- Simulated missed learner-error mass: `M = sum(w * (1 - p) * (1 - r))`.
- Successful executions per target case: `V = sum(w * r * e)`.
- Failed executions per target case: `F = C - V`.
- Estimated API fee per target case: `D = sum(w * r * d)`.
- Sequential service seconds per target case: `T = sum(w * r * t)`.

`T` is an expected sum of service times, not a concurrent workload makespan. `M` comes entirely from the simulation and must not change with the LLM configuration. Calibration estimates learner success; it does not estimate LLM confidence or repair model answers.

Predictions come from the unchanged `data/predictions.csv`: three logging settings, 200 calibration replications, six methods, plus an oracle using known probabilities. A threshold policy calls when the score is below the threshold, using 0.4, 0.6, 0.7, 0.8 and 0.9. An equal-call policy uses budgets from 0 to 1 in steps of 0.1, prioritizing lower scores and assigning the same fractional call probability to every cell in a score tie. It never uses gold labels to break ties.

Every output reports all settings. At fixed call budgets, Global and Weighted global must match Raw because they preserve its ranking. The implementation also checks the aggregated policy calculation against direct weighting of individual responses. Equal call budgets are not equal dollar budgets.

## Uncertainty and limits

Accuracy contrasts use 2,000 paired bootstrap draws with NumPy seed 20260926. Both responses to a task stay together and are averaged before resampling. Tasks are resampled within family × difficulty strata. A separate first-repeat analysis uses the same 144 exercises and reports sensitivity to repeating requests; it is not a new independent test set.

Intervals are exploratory and conditional on this synthetic construction, with no multiplicity correction. All-correct strata can produce degenerate intervals; this does not establish equivalence or zero population error. Policy comparisons are point estimates averaged across calibration replications, with no new significance claim. The experiment does not measure learning gains, real-user personalization, semantic explanation quality, or the benefit of a trained model selector.

## Reproduction

From the repository root after installing the project dependencies:

```sh
python -m fahrai_calibration.llm verify
python -m fahrai_calibration.llm analyze --output outputs/llm
```

Both commands are offline and need no credentials. Verification checks file hashes, regenerates all exercises and the schedule, regrades every final response, reconstructs usage-based fee estimates, reruns both analysis samples, and compares them with the reference CSVs. Numeric comparisons use relative tolerance 1e-10 and absolute tolerance 1e-12. The output directory must not already exist.

This release reproduces analysis of the recorded experiment. It does not include a paid-request scheduler. For new collection, `requests.jsonl` supplies the exact provider request bodies; each record is keyed by configuration and assignment. Configure credentials in the chosen API client through environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). Keep new outputs separate from the reference bundle and set an explicit spending limit before dispatch. Local Ollama requests require no cloud key. Live model access and generation are not needed to check the published findings.
