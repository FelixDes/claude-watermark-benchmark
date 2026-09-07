# claude-watermark-benchmark

Black-box detection of SynthID-Text watermarks: a reference implementation of
the Red-Green *presence* test from Gloaguen et al. (ICLR 2025), its validation
on a locally watermarked model, and a field study against Claude's production
SynthID deployment — including a controlled experiment showing how a one-line
deployment choice makes the published test blind, and two new probe designs
with their failure modes.

**Findings in brief**

- The paper's test works as described on a reference SynthID (`transformers`)
  implementation: S = 10, p ≈ 0.001 at 10 samples per cell, clean control, and
  the context size `h` is recoverable by a sweep.
- Seeding SynthID's repeated-context history with the *prompt* — a change a
  deployer has every reason to make — leaves the watermark fully active and the
  test completely blind (S = 1, p = 1.0). Reproduced locally in isolation.
- On `claude-fable-5-1`, the only Claude model Anthropic states is watermarked
  on the API, the test finds nothing across **eight configurations** and ~12 000
  well-formed responses: both transports, contexts up to 15 tokens / 6 words,
  context spelled in the prompt or constructed by the model, with and without a
  150-word paragraph before the probe. The Haiku 4.5 control is equally null,
  as expected for a model in the transition period.
- Two probe redesigns fix specific assumptions and break on others: describing
  the context instead of spelling it introduces digit-semantics false positives
  on small models; a within-response repeat design ("masking differential")
  is correct on three local ground truths and confounded on the Claude family,
  where the unwatermarked control fires as hard as the watermarked model.
- Still open for the Fable null: a hash that excludes digits, a key that varies
  across requests, a context wider than the local window, or the watermark not
  being live on this API route. A detection-API check on our own responses
  would settle it in one call.

## Contents

| file | what |
|---|---|
| `blackbox_redgreen.py` | the presence test, context sweep, context-size estimator, the three probes, CLI |
| `endpoints.py` | one sampling contract, three backends: local `transformers`, Anthropic sync, Anthropic batch |
| `watermarks.py` | `PromptAwareSynthIDProcessor` — SynthID with a prompt-seeded repeated-context history |
| `synthid_demo.py` | minimal SynthID generation + the training-free mean-g-value text detector |
| `test_blackbox.py` | offline tests against fake clients: pipeline, transports, parser, statistics (`python test_blackbox.py`) |
| `runs/` | every raw response and aggregated matrix from every run in this README ([index](runs/README.md)) |

## Setup

```bash
uv sync                      # torch 2.13+cu130, transformers 5.16, scipy, anthropic; Python 3.14
python test_blackbox.py      # no GPU, no credentials
export ANTHROPIC_API_KEY=... # for the hosted-model runs
```

Local runs were done on an RTX 5070 (12 GB). `--batch-size` is the knob that
matters there: the SynthID processor holds a `[batch, vocab, depth]` buffer and
`output_scores` a `[batch, steps, vocab]` one.

---

## 1. The presence test

Gloaguen, Jovanović, Staab, Vechev, *Black-Box Detection of Language Model
Watermarks*, ICLR 2025 ([OpenReview](https://openreview.net/forum?id=JjCVLAY0H4),
[arXiv 2405.20777](https://arxiv.org/abs/2405.20777)) — §2, App. B.1 (Alg. 1),
App. C (Alg. 2/3), App. F.4 — plus the SynthID case study in the SRI Lab post
[*Probing Google DeepMind's SynthID-Text Watermark*](https://www.sri.inf.ethz.ch/blog/probingsynthid).
Cross-checked against the authors' [reference code](https://github.com/eth-sri/watermark-detection).
Only the Red-Green test is implemented.

This answers a different question from `transformers`' `WatermarkDetector` /
`SynthIDTextWatermarkDetector` ("was *this text* watermarked?"): given only
sampling access to a chat endpoint, **is a Red-Green-family watermark deployed
behind it at all?**

### Mechanism

A Red-Green watermark biases the next-token logits as a deterministic function
of the last `h` generated tokens (the *context*) and of nothing before them
(the *prefix*). SynthID-Text is not literally Red-Green — tournament sampling
gives a variable rather than fixed bias — but the bias is still a fixed
function of the context, which is the only property the test needs.

The probe steers the model into picking one of four words after a controlled
context `t2` preceded by a controlled prefix `t1`:

```
Complete the sentence "{t1} {t2}" using only and exactly a random word from the list: [...]
```

`t2` is one digit repeated `H` times, `t1` one of ten verb phrases. Sampling
each of the 90 cells `N` times gives a 10×9 matrix `L` of estimated logits for
the most-preferred word. Rows are centred on their median (removing per-prefix
taste), cells beyond `±1.96 σ̂` are flagged Red/Green, and the statistic is the
spread of per-column flag counts (Eq. 8): under a watermark whole columns shift
together, under none the flags are scattered noise. The p-value is a Monte-Carlo
permutation test over cells — valid by exchangeability under the null — reported
as the 99% Clopper–Pearson upper bound.

Two σ̂ estimators are printed because the paper (Eq. 5, `rowvar`) and the
reference code (`colstd`) disagree. `colstd` is tighter; `rowvar` is inflated by
any genuine column structure and therefore more conservative.

### What the test needs

1. **Entropy at the choice.** A near-deterministic choice has nothing to shift;
   the run is refused as uninformative when the mean top-word share exceeds 0.9.
2. **The context inside `t2`.** If `h` exceeds the tokens in `t2` the window
   reaches the varying prefix and column consistency is gone. Hence sweeps
   over `H`.
3. **`H = h` exactly for SynthID.** With repeated-context masking (`K = 1`),
   `H > h` makes the window before the choice a repeat of the window one
   token earlier, and the watermark switches off exactly where the probe
   looks.
4. **One key across all queries.**
5. **No genuine digit → word preference in the clean model.** A 1.5B model has
   one; the control catches it.
6. **The watermark active at the choice position** — see §4.

### Tokenizers change what `H` means

Qwen emits one token per digit, so `H` digits are `H` context tokens. Claude
groups digits three per token (`count_tokens`: "333" is one, "3333" is two) and
puts a standalone space token on each side of the run, so before the chosen word
the window is `[ ][ddd]…[partial][ ] <word>`. Consequences: `h = 1` is
invisible on Claude (the one context token is the constant space); `H = 1–6`
covers `h ≤ 4` and `H = 7–15` covers `h ≤ 7`; and masking never hits the choice
position because the window ends in the unique word-space, so a Claude sweep
has no hard top edge and `h` is read from the bottom of the detected band.

`--t2-style words` (`3 33 333 3333 …`) covers a *word*-level hash up to `H`
words and a token hash up to ~`H + Σ⌈k/3⌉` tokens, with no repeating window.

### Recovering `h`

`--sweep-context 2-6` runs the test per `H`, picks the word list once, and
reads `h` off the detected band. `--estimate-context` implements the paper's
Alg. 1 (perturb a digit ahead of the run, Mood scale test for where it stops
mattering); it is ±1 noisy at nine perturbations and far more expensive.

---

## 2. Validation on a local reference

`Qwen/Qwen2.5-3B-Instruct`, SynthID with 9 keys and `ngram_len = 5` (`h = 4`).
The local backend reads the choice distribution off the score vector, which for
SynthID is exact (the processor returns watermarked log-probs), then resamples
it `N` times so the estimator noise matches the black-box path.

| run | σ̂ | S | p99 | verdict |
|---|---|---|---|---|
| SynthID, `H = 4` | colstd | 10 | 0.0005 | detected |
| SynthID, `H = 4` | rowvar | 5 | 0.0047 | detected |
| control, `H = 4` | colstd | 1 | 1.00 | clean |
| SynthID, `H = 5` | colstd | 0 | 1.00 | hidden by masking, as the blog predicts |

Sweep `H = 2–6`: detected at 3 and 4 only → `h_hat = 4` (the `H = 3` hit is a
constant space token filling the fourth slot). Alg. 1 with the paper's
consecutive rule: `h_hat = 4`; with the reference repo's baseline rule: 5.

Resampling the collected matrices: 3 samples per cell over the full grid
detects in 30/30 trials, 1 per cell in 28/30; shrinking the grid to 3×3 drops
detection to 0.63 even with clean estimates. Cell count beats samples per cell.

---

## 3. Field study: Claude

On 2026-08-14 Anthropic [announced](https://www.anthropic.com/news/claude-text-watermark)
text watermarking "a version of the SynthID-Text approach", applied to models
launched after 2026-08-02 on the API worldwide with no opt-out; earlier models
are in a transition period ([help center](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content)).
Per the Models API the only such model as of 2026-09-07 is `claude-fable-5-1`
(created 2026-08-28).

Hosted-model specifics handled in `endpoints.py`: the current model generation
rejects `temperature` (provider default sampling is used; it is stochastic —
cells split 5:1, 4:2); Fable cannot disable thinking but at `effort: low`
emits none on these prompts (`output_tokens` equals the 14-token answer); a
per-request nonce at the head of the system prompt defeats any prefix reuse
without entering the context window; refusals (`stop_reason: refusal`,
category `cyber`, on "write the digit 7 twelve times"-style wording) are
dropped rather than routed to a fallback model, which would silently mix in
unwatermarked answers.

### 3.1 `claude-haiku-4-5` — transition period, expected clean

Sweep `H = 1–15`, 10 samples per cell, 13 977 requests, parse rate 100%.
`colstd` S ≤ 2 and p99 ≥ 0.64 at every `H`; `rowvar` dips to 0.06–0.12 at
`H = 9, 10, 12` without clearing 0.05. No evidence for `2 ≤ h ≤ 7`.

### 3.2 `claude-fable-5-1` — stated watermarked

All eight configurations, chronological. `n` is samples per cell after
refusals; "S / p99" is `colstd`.

| # | probe | transport | context | word list | H | n | S | p99 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | echo | batch | spelled | peaches | 12 | 6 | 1 | 1.00 | null |
| 2 | echo | batch | spelled | apples | 6 / 12 / 24 | 13 | 1 / 1 / 0 | 0.99 / 1.00 / 1.00 | null |
| A | echo | batch | six digit *words* | apples | 6 w | 7 | 1 | 1.00 | null |
| B | echo + 90-word paragraph | batch | six digit words | apples | 6 w | 7 | 4 | 0.43 | **uninformative** — pears 97.5% |
| P1 | described (context never in prompt) | batch | constructed | apples | 12 | ≤ 8 | 0 | 1.00 | null; 76 refusals |
| S | echo | **sync** | spelled | peaches | 12 | 5 | 1 | 1.00 | null |
| B2 | echo + 150-word paragraph, `max_tokens` 1024 | batch | spelled | peaches | 12 | 6 | 1 | 1.00 | null; 0.84 bits/cell |

Echoed or constructed context intact before the chosen word in 100% of parsed
responses in every run. Per-cell entropy on Fable is lower than the marginals
suggest — the prefix largely fixes the fruit (mean top share 0.73–0.82 vs 0.65
on Haiku) — and a per-cell search over six lists found peaches/plums/cherries/
apricots best (0.96 bits at 5 samples); runs S and B2 use it.

### 3.3 Hypotheses for the Fable null

| hypothesis | status | by |
|---|---|---|
| prompt-aware repeated-context masking (§4), as the *sole* cause | rejected | P1 |
| word-level hash, `h_w ≤ 6` | rejected | A |
| token context `h ≤ 15` | rejected | A, run 2 |
| thinking rehearses the sentence → masking | rejected | `output_tokens` = answer length |
| batch path not yet watermarked | rejected | S |
| output-length gate ≤ ~200 tokens, or small-`max_tokens` gate | rejected | B2 |
| too little entropy / too few samples | addressed | per-cell list search; decisive locally at the same `n` |
| **hash excludes digits** (all probes use digit contexts) | open | word-token contexts, not run |
| key varies across requests | open | within-response designs confounded on Claude (§5) |
| context wider than the local window | open | — |
| not live on this API route | open | detection API only |

---

## 4. Why the paper's test can go blind: a controlled experiment

In `transformers`, SynthID's repeated-context history starts empty and only
ever sees *generated* tokens — not even the last prompt token. A context the
model echoes from the prompt is therefore "new" and gets watermarked. A deployer
has every reason to close that gap (a proofreading model should not stamp the
user's own text as its own), and Anthropic's description of proofreading reads
that way. `watermarks.py` makes that one change: seed the history with every
`h`-gram of the prompt.

Three probes × three ground truths, Qwen2.5-3B, `H = 4`, 10 samples/cell,
`colstd` / permutation p:

| probe | SynthID stock | clean control | SynthID + prompt-aware history |
|---|---|---|---|
| **echo** (the paper's) | detect, S = 9, p99 = 0.002 | clean, S = 1 | **blind**, S = 1, p = 1.0 |
| **described** | detect, S = 8 | **false positive**, S = 6 | detect, S = 8 |
| **masking differential** | detect, T = 35.6, p99 = 0.002 | clean, T = 8.8, p = 0.30 | detect, T = 28.5, p99 = 0.002 |

Row 1 is the mechanism in isolation: the watermark is fully active and the
published test sees nothing. Rows 2–3 are the redesigns of §5.

---

## 5. Probe designs and how they fail

**Described context** (`--probe described`). The model is told how to
*construct* the digit run — "the number made of twelve copies of the digit 3" —
so the n-gram before the choice first occurs in its own output; a prompt-aware
history cannot mask it. Cost: on a 3B model the digit becomes a salient
instruction and the *clean* model develops a digit → fruit preference the column
statistic cannot tell from a watermark (control S = 6, p99 = 0.002; `rowvar`
stays clean at the price of missing the real watermark). Needs a control on the
same model; Fable complies perfectly (12- and 24-digit runs, no thinking) but
its safety classifier refuses ~10% of these prompts as `cyber`.

**Masking differential** (`--masking-test`). Turns `K = 1` from obstacle into
fingerprint. Two probe lines in one response, the choice read on line two; in
type A line one used the same digit (context already emitted → masked), in
type B a different one (context new → watermarked). Same request, so the key
is the same by construction — this survives key rotation and needs no matrix.
Statistic: Σ over contexts of χ² on the 2×4 (type × word) table, permutation
over type labels.

Correct on all three local ground truths (row 3 above). On Claude it is
confounded:

| model | watermarked | variant | T | p99 | line-1 fruit depends on type (χ², df 3) |
|---|---|---|---|---|---|
| Fable 5.1 | yes | same | 146.6 | 0.0011 | 126.5 |
| Fable 5.1 | yes | mixed | 145.5 | 0.0011 | 58.0 |
| Haiku 4.5 | **no** | same | 80.2 | 0.0011 | 92.8 |
| Haiku 4.5 | **no** | mixed | 84.2 | 0.0011 | 11.9 |

Both models never repeat the fruit across lines (P(Y = X) = 0.00 on 3 700
responses), so line two is a near-deterministic function of line one — and line
one's fruit differs between the types although, pooled over contexts, it sits
after the same set of digits in both. What distinguishes A from B before line
one is written is the prompt: "the same number twice" vs "two different
numbers". The `mixed` variant (a mixed number on line one whose last five
digit tokens either do or do not repeat line two's window) halves the cue and
does not remove it; the residual after stratifying on line one keeps one sign
across contexts, which a context-keyed hash cannot produce. Claude models plan
the whole response from the whole prompt; a within-response repeat design is
valid on this family only if the two conditions are *literally the same
prompt* (e.g. the model picks line one's trailing digit itself, classified
post hoc). Not run.

Design lessons that generalise: a control on the same model family is
mandatory (Qwen's control was clean and hid both flaws); refuse a verdict when
the choice is near-deterministic; never fill unobserved cells with a shared
prior (clustered failures then read as column agreement — such cells are
neutralised to the row median).

---

## 6. Engineering notes

**Contract.** `Endpoint.sample_texts(prompts, n)` is all the test needs;
`choice_probs` is an optional exact fast path the local backend provides.
`LocalHFEndpoint.variant(...)` builds the control and the prompt-aware variant
over the same loaded weights.

**Anthropic transports.** `AnthropicEndpoint` (one request per sample, results
stream back; use while exploring) and `AnthropicBatchEndpoint` (Batches API:
half the price, no rate-limit juggling, one wait per submission; use for full
runs). The batch path over-requests by `--oversample` (1.3) so the retry round
is usually unnecessary, keys results by `custom_id`, and reads per-item errors
from `result.error.error.message`. `temperature` is routed through
`extra_body` on the sync path because the SDK dropped it from the typed
signature; on the batch path it serialises through the `params` dict as is.
Auth is checked before any request; 401/403/404 abort; 25 consecutive failures
trip a breaker; refusals are counted separately.

**Budget.** The planned request count and a cost estimate print before
anything is spent (cache-aware for sweeps), `--max-requests` is a hard cap, and
the run warns with a suggested `--n-samples` when the plan exceeds it. Total
spend for everything in this README: about $24 on the Anthropic API.

**What gets written** (`--data-prefix runs/x`): `x_<tag>_H<n>.npz`, the
aggregated matrix per `H` (reused on re-run, so a sweep can be re-scored for
free), and `x_responses.jsonl`, every raw response appended as it arrives:

```json
{"kind":"response","tag":"api_claude-fable-5-1","H":12,"t1":"I ate","t2":"333333333333",
 "attempt":0,"prompt":"Complete the sentence ...","response":"I ate 333333333333 plums.","parsed":"plums"}
```

---

## 7. Reproduction

```bash
# local: watermarked + control, then the masking sweep and h estimation
python blackbox_redgreen.py --batch-size 6
python blackbox_redgreen.py --batch-size 6 --sweep-context 2-6 --skip-control
python blackbox_redgreen.py --batch-size 6 --estimate-context

# the 3x3 matrix of section 4
P="python blackbox_redgreen.py --batch-size 6 --local-variants synthid,control,prompt-aware --context 4 --words peaches,plums,cherries,apricots"
$P --probe echo; $P --probe described; $P --masking-test

# hosted: cost preview needs no key; runs do
python blackbox_redgreen.py --endpoint anthropic --sweep-context 1-15 --n-samples 10 --data-prefix runs/haiku
python blackbox_redgreen.py --endpoint anthropic-batch --api-model claude-fable-5-1 \
    --words peaches,plums,cherries,apricots --context 12 --n-samples 6 --data-prefix runs/fable
python blackbox_redgreen.py --endpoint anthropic-batch --api-model claude-fable-5-1 \
    --probe described --context 12 --n-samples 6 --words apples,bananas,oranges,pears
python blackbox_redgreen.py --endpoint anthropic-batch --api-model claude-fable-5-1 \
    --preamble --preamble-words 150 --api-max-tokens 1024 --context 12 --n-samples 5 \
    --words peaches,plums,cherries,apricots
python blackbox_redgreen.py --endpoint anthropic-batch --api-model claude-fable-5-1 \
    --masking-test --masking-variant mixed --context 18 --n-samples 40 --words apples,bananas,oranges,pears
```

## References

- Gloaguen, Jovanović, Staab, Vechev. *Black-Box Detection of Language Model Watermarks.* ICLR 2025. [arXiv 2405.20777](https://arxiv.org/abs/2405.20777).
- Dathathri et al. *Scalable watermarking for identifying large language model outputs.* Nature 2024 (SynthID-Text).
- SRI Lab. [*Probing Google DeepMind's SynthID-Text Watermark.*](https://www.sri.inf.ethz.ch/blog/probingsynthid) 2024.
- Anthropic. [*How Claude's text watermarking works.*](https://www.anthropic.com/news/claude-text-watermark) 2026-08-14.
