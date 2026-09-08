# Raw data

`*_responses.jsonl`: one record per response (prompt, response, parsed word,
cell). `*.npz`: aggregated 10×9×4 matrix per `H`. Prompts and model output
only.

| prefix | model | run | README |
|---|---|---|---|
| `h2_` … `h5_` | Qwen2.5-3B | echo at fixed `H`, SynthID + control | §2 |
| `ctx_` | Qwen2.5-3B | Alg. 1 context estimation | §2 |
| `m33_` | Qwen2.5-3B | 3×3 matrix: echo / described / masking × stock / control / prompt-aware | §4 |
| `m33b_` | Qwen2.5-3B | described, mangoes list | §5 |
| `haiku-probe_` | Haiku 4.5 | first probe, `H = 12` | §3.1 |
| `haiku-sweep_` | Haiku 4.5 | sweep `H = 1–15` | §3.1 |
| `fable_` | Fable 5.1 | run 1 | §3.2 |
| `fable2_` | Fable 5.1 | run 2, `H = 6/12/24` | §3.2 |
| `fableA_` | Fable 5.1 | run A, `--t2-style words` | §3.2 |
| `fableB_` | Fable 5.1 | run B, 90-word preamble | §3.2 |
| `fableP1_` | Fable 5.1 | run P1, described | §3.2 |
| `fableSync_` | Fable 5.1 | run S, sync API | §3.2 |
| `fableB2_` | Fable 5.1 | run B2, 150-word preamble | §3.2 |
| `fableW_` | Fable 5.1 | run W, nonsense-word contexts | §3.2 |
| `fableM_`, `haikuM_` | Fable, Haiku | masking differential, `same` | §5 |
| `fableMX_`, `haikuMX_` | Fable, Haiku | masking differential, `mixed` | §5 |

Tags: `wm` / `control` / `wm-pa` local variants; `api_<model>` hosted, with
`_words`, `_pre`, `_described`, `_mask`, `_maskmix` probe suffixes.
