# claude-watermark-benchmark

Black-box detection of SynthID-Text watermarks: the Red-Green *presence* test
of Gloaguen et al. (ICLR 2025), validated on a locally watermarked model and
applied to Claude's production SynthID deployment.

**Findings**

- The test works on reference SynthID (`transformers`): S = 10, p ≈ 0.001 at
  10 samples/cell; control clean; `h` recoverable by sweep.
- Seeding SynthID's repeated-context history with the prompt leaves the
  watermark active and the test blind (S = 1, p = 1.0). Reproduced locally.
- On `claude-fable-5-1`, the only Claude model Anthropic states is watermarked
  on the API: nothing across 8 configurations, ~12 000 responses — both
  transports, contexts ≤ 15 tokens / 6 words, spelled or constructed context,
  with/without a 150-word paragraph before the probe. Haiku 4.5 (transition
  period) equally null.
- Two redesigns fix one assumption and break on another: a described context
  gives digit-semantics false positives on small models; a within-response
  "masking differential" is correct on three local ground truths and confounded
  on Claude, where the unwatermarked control fires as hard as the watermarked
  model.
- Open: hash excluding digits, per-request key, context wider than the local
  window, or watermark not live on this route. A detection-API check on our
  own responses would settle it.

## Contents

| file | what |
|---|---|
| `blackbox_redgreen.py` | presence test, context sweep, Alg. 1 estimator, three probes, CLI |
| `endpoints.py` | one sampling contract; backends: local `transformers`, Anthropic sync, Anthropic batch |
| `watermarks.py` | `PromptAwareSynthIDProcessor`: SynthID with prompt-seeded repeated-context history |
| `synthid_demo.py` | SynthID generation + mean-g-value text detector |
| `test_blackbox.py` | offline tests on fake clients (`python test_blackbox.py`) |
| `runs/` | all raw responses and matrices ([index](runs/README.md)) |

## Setup

```bash
uv sync                      # torch 2.13+cu130, transformers 5.16, scipy, anthropic; Python 3.14
python test_blackbox.py      # no GPU, no credentials
export ANTHROPIC_API_KEY=... # hosted runs
```

Local runs: RTX 5070, 12 GB. `--batch-size` bounds the SynthID
`[batch, vocab, depth]` and `output_scores` `[batch, steps, vocab]` buffers.

---

## 1. The test

Gloaguen, Jovanović, Staab, Vechev, *Black-Box Detection of Language Model
Watermarks*, ICLR 2025 ([arXiv 2405.20777](https://arxiv.org/abs/2405.20777)),
§2, App. B.1, C, F.4; SRI Lab
[SynthID post](https://www.sri.inf.ethz.ch/blog/probingsynthid);
[reference code](https://github.com/eth-sri/watermark-detection). Red-Green
test only.

Question: given sampling access, is a Red-Green-family watermark deployed at
all? (Not "is this text watermarked" — that is `SynthIDTextWatermarkDetector`.)

**Mechanism.** A Red-Green watermark biases next-token logits as a fixed
function of the last `h` tokens (context), not of what precedes them (prefix).
SynthID's tournament bias is variable but still a fixed function of context.
Probe:

```
Complete the sentence "{t1} {t2}" using only and exactly a random word from the list: [...]
```

`t2` = one digit × `H`, `t1` = one of 10 verb phrases, 4-word list. 90 cells ×
`N` samples → 10×9 logit matrix of the preferred word. Rows centred on median;
cells beyond `±1.96 σ̂` flagged Red/Green; statistic = spread of per-column flag
counts (Eq. 8). Watermark → whole columns shift; none → scattered noise.
p-value: permutation over cells, 99% Clopper–Pearson upper bound. Two σ̂:
`rowvar` (paper Eq. 5, conservative) and `colstd` (reference code, tighter).

**Requirements.** Entropy at the choice (verdict withheld above 0.9 top-word
share); context inside `t2`; `H = h` exactly for SynthID (`H > h` repeats the
window → masked); one key across queries; no genuine digit→word preference in
the clean model; watermark active at the choice position (§4).

**Tokenizers.** Qwen: one token per digit. Claude: three digits per token plus
a standalone space on each side, so the window is `[ ][ddd]…[partial][ ] <word>`.
Hence on Claude `h = 1` is invisible, `H = 1–6` covers `h ≤ 4`, `H = 7–15`
covers `h ≤ 7`, masking never hits the choice, and `h` reads from the bottom
of the detected band. `--t2-style words` (`3 33 333 …`) covers word-level
hashes up to `H` words with no repeating window.

**Recovering `h`.** `--sweep-context 2-6` (word list picked once) or
`--estimate-context` (Alg. 1, Mood test; ±1 noisy, expensive).

---

## 2. Local validation

Qwen2.5-3B-Instruct, SynthID 9 keys, `ngram_len = 5` (`h = 4`). Local backend
reads the exact watermarked distribution off the scores, then resamples `N`
times to match black-box noise.

| run | σ̂ | S | p99 | verdict |
|---|---|---|---|---|
| SynthID, `H = 4` | colstd | 10 | 0.0005 | detected |
| SynthID, `H = 4` | rowvar | 5 | 0.0047 | detected |
| control, `H = 4` | colstd | 1 | 1.00 | clean |
| SynthID, `H = 5` | colstd | 0 | 1.00 | hidden by masking |

Sweep `H = 2–6`: hits at 3, 4 → `h_hat = 4`. Alg. 1: consecutive rule 4,
baseline rule 5. Resampled power: 3 samples/cell detects 30/30, 1/cell 28/30;
a 3×3 grid drops to 0.63.

---

## 3. Field study: Claude

Anthropic, 2026-08-14: text watermark "a version of SynthID-Text", models
launched after 2026-08-02, API, worldwide, no opt-out; earlier models in
transition
([announcement](https://www.anthropic.com/news/claude-text-watermark),
[help center](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content)).
Only such model as of 2026-09-07: `claude-fable-5-1` (created 2026-08-28).

Hosted specifics: current generation rejects `temperature` (provider default,
stochastic: cells split 5:1, 4:2); Fable cannot disable thinking but emits
none at `effort: low` (`output_tokens` = 14-token answer); nonce at head of
system prompt; refusals (`stop_reason: refusal`, `cyber`, on "write the digit 7
twelve times" wording) dropped — never routed to a fallback model.

### 3.1 `claude-haiku-4-5` (transition period)

Sweep `H = 1–15`, 10/cell, 13 977 requests, parse 100%. `colstd` S ≤ 2,
p99 ≥ 0.64 at every `H`. Null for `2 ≤ h ≤ 7`.

### 3.2 `claude-fable-5-1` (stated watermarked)

`n` = samples/cell after refusals; S, p99 = `colstd`.

| # | probe | transport | context | list | H | n | S | p99 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | echo | batch | spelled | peaches | 12 | 6 | 1 | 1.00 | null |
| 2 | echo | batch | spelled | apples | 6 / 12 / 24 | 13 | 1 / 1 / 0 | 0.99 / 1.00 / 1.00 | null |
| A | echo | batch | six digit words | apples | 6 w | 7 | 1 | 1.00 | null |
| B | echo + 90-word paragraph | batch | six digit words | apples | 6 w | 7 | 4 | 0.43 | uninformative (pears 97.5%) |
| P1 | described | batch | constructed | apples | 12 | ≤ 8 | 0 | 1.00 | null; 76 refusals |
| S | echo | sync | spelled | peaches | 12 | 5 | 1 | 1.00 | null |
| B2 | echo + 150-word paragraph, `max_tokens` 1024 | batch | spelled | peaches | 12 | 6 | 1 | 1.00 | null; 0.84 bits/cell |

Context intact before the chosen word in 100% of parsed responses. Fable's
per-cell entropy is lower than marginals suggest (top share 0.73–0.82 vs 0.65
on Haiku); a per-cell search over six lists found peaches/plums/cherries/apricots
best; S and B2 use it.

### 3.3 Hypotheses

| hypothesis | status | by |
|---|---|---|
| prompt-aware masking (§4) as sole cause | rejected | P1 |
| word-level hash, `h_w ≤ 6` | rejected | A |
| token context `h ≤ 15` | rejected | A, 2 |
| thinking rehearses sentence → masking | rejected | `output_tokens` |
| batch path not watermarked | rejected | S |
| length gate ≤ ~200 tokens / small-`max_tokens` gate | rejected | B2 |
| entropy / sample size | addressed | list search; decisive locally at same `n` |
| hash excludes digits | open | word-token contexts, not run |
| per-request key | open | within-response designs confounded (§5) |
| context wider than local window | open | — |
| not live on this route | open | detection API only |

---

## 4. Controlled experiment: how the test goes blind

In `transformers`, SynthID's repeated-context history starts empty and sees
generated tokens only; a context echoed from the prompt is "new" and gets
watermarked. A deployer has reason to close that (do not stamp the user's
text). `watermarks.py` seeds the history with every prompt `h`-gram.

Qwen2.5-3B, `H = 4`, 10/cell, `colstd` / permutation p:

| probe | SynthID stock | control | SynthID + prompt-aware history |
|---|---|---|---|
| echo (paper) | S = 9, p99 = 0.002 | S = 1 | **blind**: S = 1, p = 1.0 |
| described | S = 8 | **false positive**: S = 6 | S = 8 |
| masking differential | T = 35.6, p99 = 0.002 | T = 8.8, p = 0.30 | T = 28.5, p99 = 0.002 |

---

## 5. Probe redesigns

**Described** (`--probe described`): "the number made of twelve copies of the
digit 3" — context first occurs in the model's output, prompt-aware history
cannot mask it. Fails on small models: the clean model develops a digit→fruit
preference (control S = 6). Control on the same model mandatory. Fable
complies; its classifier refuses ~10% as `cyber`.

**Masking differential** (`--masking-test`): two lines in one response, choice
read on line two; type A repeats line one's digit (context masked), type B
does not (watermarked). Same request → same key; no matrix. Statistic: Σ χ²
over contexts on type × word, permutation over labels. Correct locally (§4,
row 3). Confounded on Claude:

| model | watermarked | variant | T | p99 | line-1 fruit vs type (χ², df 3) |
|---|---|---|---|---|---|
| Fable 5.1 | yes | same | 146.6 | 0.0011 | 126.5 |
| Fable 5.1 | yes | mixed | 145.5 | 0.0011 | 58.0 |
| Haiku 4.5 | no | same | 80.2 | 0.0011 | 92.8 |
| Haiku 4.5 | no | mixed | 84.2 | 0.0011 | 11.9 |

Both models never repeat the fruit across lines (P(Y = X) = 0.00, 3 700
responses); line one's fruit depends on the type though it follows the same
digits pooled — the model reads "same number twice" vs "different numbers"
from the prompt. The `mixed` variant halves the cue, does not remove it; the
stratified residual keeps one sign across contexts, which a context-keyed hash
cannot. Valid on Claude only with literally identical prompts (model picks the
trailing digit itself, classified post hoc). Not run.

Lessons: control on the same model family; withhold verdicts at near-zero
entropy; neutralise unobserved cells to the row median, never a shared prior.

---

## 6. Engineering

`Endpoint.sample_texts(prompts, n)` is the contract; `choice_probs` is the
local exact fast path; `.variant()` builds control / prompt-aware over shared
weights. `AnthropicEndpoint` streams results; `AnthropicBatchEndpoint` is half
price, over-requests by `--oversample` 1.3, keys results by `custom_id`.
`temperature` goes via `extra_body` (sync) or the `params` dict (batch). Auth
checked before any request; 401/403/404 abort; 25 consecutive failures trip a
breaker; refusals counted separately. Cost preview before spending,
cache-aware; `--max-requests` hard cap. Total spend here: ~$24.

`--data-prefix runs/x` writes `x_<tag>_H<n>.npz` (matrix, reused on re-run)
and `x_responses.jsonl`:

```json
{"kind":"response","tag":"api_claude-fable-5-1","H":12,"t1":"I ate","t2":"333333333333",
 "attempt":0,"prompt":"...","response":"I ate 333333333333 plums.","parsed":"plums"}
```

---

## 7. Reproduction

```bash
python blackbox_redgreen.py --batch-size 6                                   # local wm + control
python blackbox_redgreen.py --batch-size 6 --sweep-context 2-6 --skip-control
python blackbox_redgreen.py --batch-size 6 --estimate-context

P="python blackbox_redgreen.py --batch-size 6 --local-variants synthid,control,prompt-aware --context 4 --words peaches,plums,cherries,apricots"
$P --probe echo; $P --probe described; $P --masking-test                     # §4 matrix

python blackbox_redgreen.py --endpoint anthropic --sweep-context 1-15 --n-samples 10 --data-prefix runs/haiku
python blackbox_redgreen.py --endpoint anthropic-batch --api-model claude-fable-5-1 \
    --words peaches,plums,cherries,apricots --context 12 --n-samples 6
python blackbox_redgreen.py --endpoint anthropic-batch --api-model claude-fable-5-1 \
    --probe described --context 12 --n-samples 6 --words apples,bananas,oranges,pears
python blackbox_redgreen.py --endpoint anthropic-batch --api-model claude-fable-5-1 \
    --preamble --preamble-words 150 --api-max-tokens 1024 --context 12 --n-samples 5 \
    --words peaches,plums,cherries,apricots
python blackbox_redgreen.py --endpoint anthropic-batch --api-model claude-fable-5-1 \
    --masking-test --masking-variant mixed --context 18 --n-samples 40 --words apples,bananas,oranges,pears
```

## References

- Gloaguen, Jovanović, Staab, Vechev. *Black-Box Detection of Language Model Watermarks.* ICLR 2025. [arXiv 2405.20777](https://arxiv.org/abs/2405.20777)
- Dathathri et al. *Scalable watermarking for identifying large language model outputs.* Nature 2024.
- SRI Lab. [*Probing Google DeepMind's SynthID-Text Watermark.*](https://www.sri.inf.ethz.ch/blog/probingsynthid) 2024.
- Anthropic. [*How Claude's text watermarking works.*](https://www.anthropic.com/news/claude-text-watermark) 2026-08-14.

## License

MIT.
