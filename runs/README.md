# Raw data

Every run in the top-level README. `*_responses.jsonl` holds one record per
model response (prompt, response, parsed word, cell coordinates), appended as
it arrived; `*.npz` holds the aggregated 10×9×4 probability matrix per `H`
that the statistic was computed from. Prompts and model output only — no
credentials, no user data.

| prefix | model | what | README section |
|---|---|---|---|
| `h2_`, `h3_`, `h4_`, `h5_` | Qwen2.5-3B, local | echo probe at fixed `H`, SynthID and control | §2 |
| `ctx_` | Qwen2.5-3B, local | Alg. 1 context-size estimation data | §2 |
| `m33_` | Qwen2.5-3B, local | 3×3 matrix: echo / described / masking × stock / control / prompt-aware | §4 |
| `m33b_` | Qwen2.5-3B, local | described probe re-run with the mangoes list | §5 |
| `haiku-probe_` | Haiku 4.5 | first live probe, `H = 12`, 5/cell | §3.1 |
| `haiku-sweep_` | Haiku 4.5 | sweep `H = 1–15`, 10/cell | §3.1 |
| `fable_` | Fable 5.1 | run 1: echo, peaches list, `H = 12` | §3.2 |
| `fable2_` | Fable 5.1 | run 2: echo, apples list, `H = 6, 12, 24` | §3.2 |
| `fableA_` | Fable 5.1 | run A: `--t2-style words` | §3.2 |
| `fableB_` | Fable 5.1 | run B: words + 90-word preamble (uninformative) | §3.2 |
| `fableP1_` | Fable 5.1 | run P1: described context | §3.2 |
| `fableSync_` | Fable 5.1 | run S: echo over the synchronous API | §3.2 |
| `fableB2_` | Fable 5.1 | run B2: 150-word preamble, `max_tokens` 1024 | §3.2 |
| `fableM_`, `haikuM_` | Fable 5.1, Haiku 4.5 | masking differential, `same` variant | §5 |
| `fableMX_`, `haikuMX_` | Fable 5.1, Haiku 4.5 | masking differential, `mixed` variant | §5 |

Cache tags inside file names: `wm` / `control` / `wm-pa` are the local
variants; `api_<model>` is a hosted model, with `_words`, `_pre`, `_described`,
`_mask`, `_maskmix` suffixes for the probe variants.
