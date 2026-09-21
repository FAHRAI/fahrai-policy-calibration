# LLM results

Each configuration answered 144 synthetic exercises twice. The 12 development exercises are excluded.

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

Success requires the correct integer and the specified JSON structure, including a nonempty explanation. Citation validity is reported separately. Explanation quality and the prompt's word limit are outside this score.

These results describe the recorded configurations and task set. Perfect observed accuracy does not establish equivalence. Model settings, hardware and concurrency differ; the protocol records those differences.

The [reference tables](../data/llm/reference/) contain API fee estimates, latency, paired accuracy intervals and policy replay for every setting. Local API fees are zero; total compute cost was not measured. Learner-error mass in the replay is simulated.

See the [protocol](llm_experiment.md) and [data dictionary](llm_data_dictionary.md) for definitions.
