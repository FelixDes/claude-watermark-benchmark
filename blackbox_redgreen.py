"""Black-box detection of a Red-Green LLM watermark, applied to SynthID-Text.

Implements the Red-Green presence test of

    Gloaguen, Jovanovic, Staab, Vechev,
    "Black-Box Detection of Language Model Watermarks", ICLR 2025
    https://openreview.net/forum?id=JjCVLAY0H4

and its SynthID-Text case study from

    https://www.sri.inf.ethz.ch/blog/probingsynthid

This is *not* the "was this text watermarked" detector that transformers ships
(`WatermarkDetector`, `SynthIDTextWatermarkDetector`). It answers a different
question: given only sampling access to a chat endpoint, is a Red-Green-family
watermark deployed behind it at all? Neither transformers nor synthid-text has
anything for that, so the test is implemented here.

Idea (paper Sec. 2). A Red-Green watermark applies a logit bias that depends on
the last h generated tokens (the *context*) and not on what came before it (the
*prefix*). So we steer the model into picking one word out of a small set S, in
a sentence whose context is a controlled token t2 and whose prefix is a
controlled string t1:

    Complete the sentence "{t1} {t2}" using a random word from: [{S}]

with t2 = a single digit repeated H times. Sampling many completions per
(t1, t2) cell gives a matrix L[t1, t2] of estimated logits for one chosen word
x*. Under no watermark, deviations in L are model noise and independent of t2.
Under a Red-Green watermark, whole *columns* of L shift together, because the
red/green split is a function of t2 alone. A permutation test on that
column-consistency is the p-value.

SynthID-Text is not literally a Red-Green scheme -- tournament sampling gives a
variable, not fixed, bias -- but the bias is still a deterministic function of
the preceding context, which is the only property the test needs. The one
SynthID-specific adjustment (blog, Sec. 1) is that H must equal h *exactly*
rather than merely upper-bound it: SynthID masks repeated contexts (K=1), so
H > h makes the n-gram before the choice a repeat and silently disables the
watermark at the position we are probing. For ngram_len=5 the context is the
preceding 4 tokens, hence H=4. `--estimate-context` recovers h on its own via
the Mood-test procedure of App. B.1.

Backends live in `endpoints.py` behind one contract, so the same test runs
against a local watermarked checkpoint (with a known-clean control to calibrate
against) or against a hosted model over the API (text only, no control).

Usage:
    python blackbox_redgreen.py                  # watermarked + control run
    python blackbox_redgreen.py --estimate-context
    python blackbox_redgreen.py --slow           # no logit peeking at all
    python blackbox_redgreen.py --endpoint anthropic --n-samples 25
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import numpy as np
from scipy.stats import beta, mood

from endpoints import (
    AnthropicBatchEndpoint,
    AnthropicEndpoint,
    Endpoint,
    LocalHFEndpoint,
)

PREAMBLE_SYSTEM = (
    "Follow the formatting instructions exactly: first the paragraph, then the "
    "completed sentence on its own final line. Nothing else -- no headings, no "
    "commentary, no quotation marks around the sentence."
)

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
WATERMARK_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]
NGRAM_LEN = 5

# t1: the prefix. Ten of them, following the reference implementation.
PREFIXES = [
    "I ate", "I chose", "I picked", "I selected", "I took",
    "I went for", "I settled on", "I got", "I gathered", "I harvested",
]
# t2: a single digit token, repeated H times. Nine of them.
DIGITS = list(range(1, 10))

# Candidate word lists. Phase 1 picks whichever keeps the model from collapsing
# onto one answer; `example` is used in the prompt's format demo so the demo
# itself does not bias the choice.
WORD_LISTS = [
    (["peaches", "plums", "cherries", "apricots"], "strawberries"),
    (["mangoes", "pineapples", "papayas", "kiwis"], "strawberries"),
    (["strawberries", "blueberries", "raspberries", "blackberries"], "apples"),
    (["apples", "bananas", "oranges", "pears"], "strawberries"),
]

R_THRESHOLD = 1.96
#: Above this mean top-word probability the matrix is treated as degenerate.
DEGENERATE_TOP_SHARE = 0.9


#: How the controlled context t2 is spelled out of one digit d and a size H.
#:   digits -- "3333": one word of H digits. On a per-digit tokenizer that is
#:             H tokens; on Claude it is ceil(H/3) tokens and *one* word.
#:   words  -- "3 33 333 3333": H words of distinct lengths. Covers a hash over
#:             words up to H deep and over tokens up to ~H + sum(ceil(k/3))
#:             deep, and no n-gram window ever repeats, so repeated-context
#:             masking cannot hit the choice position.
T2_STYLES = ("digits", "words")


def t2_string(d, H: int, style: str = "digits") -> str:
    if style == "words":
        return " ".join(str(d) * k for k in range(1, H + 1))
    return str(d) * H


#: For the output-length probe: a neutral topic with no fruit and no digits, so
#: the paragraph cannot collide with the parser or with the context window.
PREAMBLE_TOPIC = "how the gears on a bicycle let a rider climb a hill"


_NUM_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve "
              "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
              "twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six "
              "twenty-seven twenty-eight twenty-nine thirty").split()


def num_word(n: int) -> str:
    """Spell a small count so the prompt carries no digit run of its own."""
    return _NUM_WORDS[n] if n < len(_NUM_WORDS) else str(n)


PROBES = ("echo", "described")


def build_described_prompt(prefix: str, d: str, H: int, words: list[str]) -> str:
    """The probe whose context never appears in the prompt.

    The model is told how to *construct* the digit run rather than shown it, so
    the n-gram before the choice occurs for the first time in the model's own
    output. Under a repeated-context history that includes the prompt, the echo
    probe is masked at exactly the position it measures; this one is not. The
    format line uses placeholders for the same reason.
    """
    # "the number made of N copies of the digit d" -- not "the digit d written
    # N times in a row": Fable's safety classifier refuses the latter as
    # cyber content about one time in ten (and the two-line form nearly
    # always), the former it never did in 8/8 probes.
    return (
        f'Write "{prefix}", a space, the number made of {num_word(H)} copies of '
        f"the digit {d}, a space, and one word chosen at random from the list: "
        f"{words}. Reply with exactly that one line, ending in a period, and "
        f"nothing else. Format: {prefix} <number> <word>."
    )


def build_two_line_prompt(prefix: str, d1: str, d2: str, H: int, words: list[str]) -> str:
    """Two probe sentences in one response, for the masking-differential test."""
    n = num_word(H)
    return (
        f'Write two lines. Line one: "{prefix}", a space, the number made of {n} '
        f"copies of the digit {d1}, a space, and a random fruit from {words}. "
        f'Line two: "{prefix}", a space, the number made of {n} copies of the '
        f"digit {d2}, a space, and a random fruit from {words}, chosen "
        f"independently. Each line ends with a period. Nothing else."
    )


def build_two_line_mixed_prompt(prefix: str, head: str, tail: str, c: str, H: int,
                                words: list[str]) -> str:
    """Masking-differential prompt without the "same number twice" cue.

    Line one is a *mixed* number in both types: `head` x3 then `tail` x(H-3).
    Line two is `c` x H. In type A tail == c, so the last few digit tokens
    before line one's word are the same as before line two's word -- the
    context repeats even though the two numbers differ; in type B tail != c
    and it does not. Neither type shows the model an identical number twice.
    """
    return (
        f'Write two lines. Line one: "{prefix}", a space, the number made of '
        f"three copies of the digit {head} followed by {num_word(H - 3)} copies "
        f"of the digit {tail}, a space, and a random fruit from {words}. "
        f'Line two: "{prefix}", a space, the number made of {num_word(H)} copies '
        f"of the digit {c}, a space, and a random fruit from {words}, chosen "
        f"independently. Each line ends with a period. Nothing else."
    )


PREAMBLE_WORDS = 80


def build_prompt(prefix: str, k: str, words: list[str], example: str,
                 preamble: bool = False, probe: str = "echo") -> str:
    if probe == "described":
        # k is the intended digit run; recover (d, H) from it.
        return build_described_prompt(prefix, k[0], len(k), words)
    core = (
        f'Complete the sentence "{prefix} {k}" using only and exactly a random '
        f"word from the list: {words}. Answer in this specific format: "
        f"{prefix} {k} {example}. (the example uses a different fruit on "
        f"purpose, you have to choose among {words})"
    )
    if not preamble:
        return core
    # The sentence must come *after* a stretch of generated text, so that a
    # watermark which only switches on past some output length is active by
    # the time the choice is made. The paragraph precedes the digits, so it
    # stays outside the context window at the choice position.
    return (
        f"First write one paragraph of about {PREAMBLE_WORDS} words explaining "
        f"{PREAMBLE_TOPIC}. Do not mention any fruit or write any digits in "
        f"that paragraph. Then, on a new line, {core[0].lower()}{core[1:]}"
    )


def identify_word(text: str, words: list[str], t2: str | None = None) -> int | None:
    """Which candidate the model chose.

    With `t2` given, look at the text right after the echoed context first --
    that is where the choice is, and anything the model wrote before the
    sentence cannot confuse it. Otherwise (or if the echo is missing) fall back
    to the reference rule: exactly one candidate present, exactly once.
    """
    if t2 is not None:
        pos = text.rfind(t2)
        if pos >= 0:
            tail = text[pos + len(t2):].lstrip()
            hits = [(tail.find(w), i) for i, w in enumerate(words) if w in tail]
            if hits:
                return min(hits)[1]
    found = [(i, text.count(w)) for i, w in enumerate(words) if text.count(w) > 0]
    if len(found) == 1 and found[0][1] == 1:
        return found[0][0]
    return None


class ResponseLog:
    """Append-only JSONL record of everything the endpoint returned.

    Written as data arrives rather than at the end, so a run killed halfway --
    budget cap, rate limits, Ctrl-C -- still leaves usable evidence behind. The
    aggregated matrices are recoverable from this file; the reverse is not true.
    """

    def __init__(self, path: str | None):
        self.path = path
        self.n = 0
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, **record):
        if not self.path:
            return
        self.n += 1
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_matrix(ep: Endpoint, words, example, H, n_samples, fast, retries, verbose,
                   log: "ResponseLog | None" = None, tag: str = "",
                   t2_style: str = "digits", preamble: bool = False,
                   probe: str = "echo"):
    """Estimate the choice distribution for every (t1, t2) cell.

    Returns an array of shape (len(PREFIXES), len(DIGITS), len(words)). Counts
    start at 1 per word (Laplace smoothing) so the logit is always finite.

    Two paths, same estimator. With `fast` the endpoint hands over the exact
    choice distribution and we draw `n_samples` observations from it -- identical
    multinomial noise to the black-box path, at one generation per cell instead
    of a hundred. Without it we parse sampled text, which is all a hosted model
    allows.
    """
    cells = [(p, d) for p in PREFIXES for d in DIGITS]
    t2s = [t2_string(d, H, t2_style) for _, d in cells]
    prompts = [build_prompt(p, t2, words, example, preamble, probe)
               for (p, _), t2 in zip(cells, t2s)]
    digits = [str(d) for _, d in cells]
    if fast and t2_style != "digits":
        raise ValueError("the local fast path locates a single digit run; "
                         "use --slow with --t2-style words")
    counts = [np.ones(len(words)) for _ in cells]

    pending = list(range(len(cells)))
    for attempt in range(retries):
        if not pending:
            break
        sub = [prompts[i] for i in pending]
        still = []

        if fast:
            probes = ep.choice_probs(sub, [digits[i] for i in pending], H, words)
            for i, res in zip(pending, probes):
                if res is None:
                    still.append(i)
                    continue
                probs, mass = res
                draws = np.random.choice(len(words), size=n_samples, p=probs)
                counts[i] += np.bincount(draws, minlength=len(words))
                if log:
                    log.write(kind="probe", tag=tag, H=H, t1=cells[i][0],
                              t2=t2s[i], words=list(words),
                              probs=[float(x) for x in probs], mass=float(mass))
                if verbose:
                    print(f"  cell {i:3d} mass={mass:.3f} p={np.round(probs, 3)}")
        else:
            for i, texts in zip(pending, ep.sample_texts(sub, n_samples)):
                got = 0
                for text in texts:
                    w = identify_word(text, words, t2s[i])
                    if w is not None:
                        counts[i][w] += 1
                        got += 1
                    if log:
                        log.write(kind="response", tag=tag, H=H, t1=cells[i][0],
                                  t2=t2s[i], attempt=attempt,
                                  prompt=prompts[i], response=text,
                                  parsed=None if w is None else words[w])
                if got < max(1, n_samples // 2):
                    still.append(i)
                if verbose:
                    print(f"  cell {i:3d} parsed {got}/{n_samples} -> {counts[i].astype(int)}")

        if still and verbose:
            print(f"  retry {attempt + 1}: {len(still)} cells under-sampled")
        pending = still

    if pending:
        print(f"  warning: {len(pending)} cells stayed under-sampled; "
              "their estimates lean on the smoothing prior")

    data = np.full((len(PREFIXES), len(DIGITS), len(words)), np.nan)
    for i, (prefix, d) in enumerate(cells):
        if counts[i].sum() > len(words):  # at least one real observation
            data[PREFIXES.index(prefix), DIGITS.index(d)] = counts[i] / counts[i].sum()
    return data


def logit(p):
    return np.log(p / (1.0 - p))


def red_green_statistic(L: np.ndarray, r: float = R_THRESHOLD, sigma_mode: str = "rowvar") -> int:
    """S(L) from paper Eq. 8 / Alg. 3, on the logit matrix of the chosen word.

    Center each row (fixed t1) on its own median so per-prefix model bias drops
    out, then flag cells more than r sigma below (Red) or above (Green) it. The
    statistic is the spread in per-column flag counts: large only if some
    contexts are flagged far more consistently than others, which is what a
    context-keyed logit bias produces and plain model noise does not.

    Two sigma estimators, because the paper and the authors' code disagree:
      rowvar  -- Eq. 5, median over t1 of Var_{t2}(L). Conservative: a genuine
                 t2-dependence of the model inflates it, so it resists the false
                 positives that small models produce (they really do associate
                 particular digits with particular words).
      colstd  -- median over t2 of Std_{t1}(L), as in the reference repo. The
                 watermark shift is constant down a column so this spread is
                 watermark-free, which makes it tighter -- and more trigger-happy.
    """
    median = np.median(L, axis=1, keepdims=True)          # per-t1
    if sigma_mode == "rowvar":
        sigma = float(np.sqrt(np.median(np.var(L, axis=1))))
    else:
        sigma = float(np.median(np.std(L, axis=0)))
    centered = L - median
    red = (centered < -r * sigma).sum(axis=0)             # per-t2 counts
    green = (centered > r * sigma).sum(axis=0)
    return int(max(red.max(), green.max()) - max(red.min(), green.min()))


def permutation_test(data: np.ndarray, n_permutations: int, rng: np.random.Generator,
                     sigma_mode: str = "rowvar"):
    """Monte-Carlo permutation test over cells; returns (S, p, p_upper_99)."""
    n1, n2, n_words = data.shape
    L_all = logit(data)
    chosen = int(np.argmax(np.nansum(L_all, axis=(0, 1))))
    # A cell with no observations must not vote. Filling it with the uniform
    # prior gives every such cell the same logit, and if failures cluster on a
    # digit that fake agreement is flagged as a column shift. The row median is
    # the one value guaranteed to sit inside the flag band.
    missing = np.isnan(L_all[:, :, chosen])
    if missing.any():
        row_med = np.nanmedian(L_all, axis=1, keepdims=True)
        L_all = np.where(np.isnan(L_all), np.broadcast_to(row_med, L_all.shape), L_all)
        print(f"  note: {int(missing.sum())} cells had no observations; neutralised")
    L = L_all[:, :, chosen]

    observed = red_green_statistic(L, sigma_mode=sigma_mode)
    flat = L_all.reshape(-1, n_words)
    hits = 0
    for _ in range(n_permutations):
        perm = rng.permutation(flat).reshape(n1, n2, n_words)[:, :, chosen]
        hits += red_green_statistic(perm, sigma_mode=sigma_mode) >= observed

    p = hits / n_permutations
    # Upper bound of the 99% Clopper-Pearson interval, as the paper reports.
    p_upper = 1.0 if hits == n_permutations else beta.ppf(
        0.995, hits + 1, n_permutations - hits
    )
    return observed, p, float(p_upper), chosen


def pick_word_list(ep: Endpoint, H, probe_samples, verbose, forced=None,
                   t2_style: str = "digits", preamble: bool = False,
                   probe: str = "echo"):
    """Phase 1 of the paper: find a word set the model does not collapse onto.

    `forced=(words, example)` skips the probe entirely -- for when a previous
    run already showed which list this model spreads over.

    Text-only, so it runs identically against a local model and a hosted one.
    A list is usable when the model both follows the format and spreads its
    answers -- with no entropy at the choice position there is nothing for a
    watermark to shift, and the matrix is constant.
    """
    if forced:
        return forced
    for words, example in WORD_LISTS:
        pairs = list(zip(PREFIXES[:probe_samples], DIGITS[:probe_samples]))
        prompts = [build_prompt(p, t2_string(d, H, t2_style), words, example,
                                preamble, probe)
                   for p, d in pairs]
        picks = [
            identify_word(t, words, t2_string(d, H, t2_style))
            for (_, d), texts in zip(pairs, ep.sample_texts(prompts, 1)) for t in texts
        ]
        valid = [p for p in picks if p is not None]
        if not valid:
            if verbose:
                print(f"  {words}: unparseable, skipping")
            continue
        top = max(np.bincount(valid, minlength=len(words))) / len(valid)
        if verbose:
            print(f"  {words}: parse rate {len(valid)}/{len(prompts)}, top share {top:.2f}")
        if len(valid) >= 0.5 * len(prompts) and top <= 0.8:
            return words, example
    print("  no word list passed the diversity check; using the first one")
    return WORD_LISTS[0]


def response_log(args) -> ResponseLog:
    return ResponseLog(
        f"{args.data_prefix}_responses.jsonl" if args.data_prefix else None
    )


def run_test(ep, args, label, tag, words_example=None, log=None, quiet=False):
    if not quiet:
        print(f"\n### {label}")
    log = log or response_log(args)
    cache = f"{args.data_prefix}_{tag}.npz" if args.data_prefix else None
    if cache and os.path.exists(cache):
        blob = np.load(cache, allow_pickle=True)
        data, words = blob["data"], list(blob["words"])
        print(f"  reusing collected data from {cache}")
    else:
        words, example = words_example or pick_word_list(
            ep, args.context, len(PREFIXES), args.verbose, forced_words(args),
            args.t2_style, args.preamble, args.probe)
        if not quiet:
            print(f"  word list: {words} (example: {example}), H={args.context}")
        data = collect_matrix(
            ep, words, example, args.context, args.n_samples,
            use_fast_path(ep, args), args.retries, args.verbose, log, tag,
            args.t2_style, args.preamble, args.probe,
        )
        if cache:
            np.savez(cache, data=data, words=np.array(words))

    # A near-deterministic choice has nothing for a watermark to shift, and in
    # logit space its tiny fluctuations look enormous against a tiny sigma.
    # Neither "detected" nor "no evidence" means anything there.
    observed = ~np.isnan(data[:, :, 0])
    top_share = float(np.nanmean(np.nanmax(data, axis=-1))) if observed.any() else 1.0
    degenerate = top_share > DEGENERATE_TOP_SHARE
    if degenerate and not quiet:
        print(f"  warning: mean top-word share {top_share:.2f} > "
              f"{DEGENERATE_TOP_SHARE}; the choice is near-deterministic and the "
              f"verdict below is withheld as uninformative")

    verdicts = {}
    for sigma_mode in ("rowvar", "colstd"):
        rng = np.random.default_rng(args.seed)
        S, p, p_upper, chosen = permutation_test(
            data, args.permutations, rng, sigma_mode
        )
        if degenerate:
            p_upper = float("nan")
        verdicts[sigma_mode] = (S, p, p_upper)
        if not quiet:
            verdict = ("UNINFORMATIVE (no entropy)" if degenerate else
                       "WATERMARK DETECTED" if p_upper < 0.05 else "no evidence")
            print(f"  sigma={sigma_mode:7s} x*={words[chosen]:<12s} S={S:2d} "
                  f"p={p:.4f} p99={p_upper:.4f}  ->  {verdict}")
    return data, verdicts, words


def estimate_context_size(ep, args, label, tag):
    """App. B.1 / Alg. 1: recover h by watching when a perturbation stops mattering.

    The prompt becomes "{t1} {d'}{d*H}". While H < h the perturbation digit d'
    sits inside the watermark context, so varying it moves the logits around;
    once H reaches h it falls out of the window and that spread collapses. A
    Mood two-sample scale test finds the collapse point.

    Two decision rules, again because paper and code differ: "consecutive"
    compares H against H-1 and fires on the abrupt change itself (Alg. 1);
    "baseline" compares every H against H=1, as the reference repo does.
    """
    print(f"\n### context-size estimation ({label})")
    cache = f"{args.data_prefix}_ctx_{tag}.npz" if args.data_prefix else None
    if cache and os.path.exists(cache):
        blob = np.load(cache, allow_pickle=True)
        data, words = blob["data"], list(blob["words"])
        print(f"  reusing collected data from {cache}")
    else:
        words, example = pick_word_list(ep, args.context, len(PREFIXES), args.verbose)
        prefix = PREFIXES[0]
        perturbations = list(range(1, 10))
        fast = use_fast_path(ep, args)

        # data[H-1, d', d, word], ones for Laplace smoothing
        data = np.ones((args.h_max, len(perturbations), len(DIGITS), len(words)))
        jobs = [
            (H, pi, di) for H in range(1, args.h_max + 1)
            for pi in range(len(perturbations)) for di in range(len(DIGITS))
        ]
        # One H at a time: the probe's digit-run length is what varies here, and
        # both endpoint paths take a single H per call.
        for H in range(1, args.h_max + 1):
            cells = [(pi, di) for pi in range(len(perturbations))
                     for di in range(len(DIGITS))]
            prompts = [
                build_prompt(prefix, f"{perturbations[pi]}{str(DIGITS[di]) * H}",
                             words, example)
                for pi, di in cells
            ]
            if fast:
                probes = ep.choice_probs(
                    prompts, [str(DIGITS[di]) for _, di in cells], H, words)
                for (pi, di), res in zip(cells, probes):
                    if res is None:
                        continue
                    draws = np.random.choice(len(words), size=args.n_samples, p=res[0])
                    data[H - 1, pi, di] += np.bincount(draws, minlength=len(words))
            else:
                for (pi, di), texts in zip(cells, ep.sample_texts(prompts, args.n_samples)):
                    for text in texts:
                        w = identify_word(text, words)
                        if w is not None:
                            data[H - 1, pi, di, w] += 1
        if cache:
            np.savez(cache, data=data, words=np.array(words))

    probs = data / data.sum(axis=-1, keepdims=True)
    choice = int(np.argmax(probs.sum(axis=(0, 1, 2))))
    L = logit(probs)[:, :, :, choice]  # [H, d', d]

    h_max = L.shape[0]
    for rule in ("consecutive", "baseline"):
        estimates = []
        for di in range(L.shape[2]):
            for H in range(2, h_max + 1):
                ref = L[H - 2, :, di] if rule == "consecutive" else L[0, :, di]
                if mood(ref, L[H - 1, :, di], alternative="greater").pvalue < 0.05:
                    estimates.append(H)
                    break
        if not estimates:
            print(f"  {rule:11s}: no collapse detected (h > h_max, or the probe failed)")
            continue
        print(f"  {rule:11s}: per-context estimates {estimates} -> "
              f"h_hat = {int(statistics.median(estimates))}")
    print(f"  (SynthID ngram_len = {NGRAM_LEN} implies h = {NGRAM_LEN - 1})")


def parse_h_range(spec: str) -> list[int]:
    """"2-6" or "2,3,5" -> [2,3,4,5,6] / [2,3,5]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def sweep_context(ep: Endpoint, args, label, tag):
    """Run the presence test at every candidate H and read h off the result.

    Detection fires only inside a narrow band of H, and the band has a hard top
    edge: at H = h+1 the n-gram before the choice position is a repeat, SynthID's
    repeated-context masking switches the watermark off exactly where we probe,
    and the signal vanishes. The bottom edge is softer -- below h the watermark's
    context window reaches back into the prompt template, and whether that breaks
    the test depends on whether the tokens it reaches are constant (harmless) or
    vary with t1 (fatal). So the top of the detected band is the estimate.
    """
    print(f"\n### context sweep ({label})")
    Hs = parse_h_range(args.sweep_context)
    log = response_log(args)

    # Pick the word list once, at the middle H, and reuse it everywhere: the
    # columns have to be comparable across H, and re-picking would also re-spend
    # the query budget once per H.
    probe_args = argparse.Namespace(**vars(args))
    probe_args.context = Hs[len(Hs) // 2]
    words, example = pick_word_list(ep, probe_args.context, len(PREFIXES),
                                    args.verbose, forced_words(args), args.t2_style,
                                    args.preamble, args.probe)
    print(f"  word list: {words} (example: {example})")
    print(f"  {'H':>3} | {'S':>3} {'p99':>8} (rowvar) | {'S':>3} {'p99':>8} (colstd)")

    detected = []
    for H in Hs:
        per_h = argparse.Namespace(**vars(args))
        per_h.context = H
        _, verdicts, _ = run_test(
            ep, per_h, label, f"{tag}_H{H}", (words, example), log, quiet=True
        )
        (s_r, _, p_r), (s_c, _, p_c) = verdicts["rowvar"], verdicts["colstd"]
        hit = bool(p_c < 0.05)  # NaN (degenerate) compares False
        print(f"  {H:>3} | {s_r:>3} {p_r:>8.4f}           | {s_c:>3} {p_c:>8.4f}"
              f"           {'<- detected' if hit else ''}")
        if hit:
            detected.append(H)

    print()
    if not detected:
        print("  no H detected a watermark: either the endpoint is unwatermarked, "
              "or h lies outside the swept range.")
    else:
        band = f"{min(detected)}-{max(detected)}" if len(detected) > 1 else str(detected[0])
        print(f"  detected band H = {band}  ->  h_hat = {max(detected)}")
        if max(detected) == max(Hs):
            print("  warning: the band runs to the top of the swept range, so its "
                  "upper edge was never observed. Extend --sweep-context upward "
                  "before trusting h_hat.")
    if log.path and log.n:
        print(f"  raw responses: {log.path} ({log.n} records)")
    elif log.path:
        print(f"  no new queries: every H reused a cached matrix from "
              f"{args.data_prefix}_*.npz")
    return detected


def forced_words(args):
    if not getattr(args, "words", None):
        return None
    words = [w.strip() for w in args.words.split(",")]
    if len(words) != 4:
        raise SystemExit("--words needs exactly four comma-separated words")
    return words, args.example


def masking_statistic(samples: dict) -> float:
    """Sum over contexts of chi-square on the 2 x |words| table (type x word)."""
    total = 0.0
    for c, (a, b) in samples.items():
        table = np.array([a, b], dtype=float) + 0.5  # Haldane smoothing
        expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
        total += float(((table - expected) ** 2 / expected).sum())
    return total


def masking_test(ep: Endpoint, args, label, tag):
    """Turn SynthID's K=1 rule into the signal instead of the obstacle.

    Two lines in one response, the choice read on line two:

        type A:  <t1> cccc <X>.   <t1> cccc <Y>.     context c seen on line 1 -> masked
        type B:  <t1> dddd <X>.   <t1> cccc <Y>.     context c new on line 2   -> watermarked

    Same position, same anti-repetition dynamics; the only difference is
    whether the model has already emitted context c. Without a watermark the
    distribution of Y is the same in A and B. With one, B is biased by the
    g-values of c and A is not. Both lines are in one request, so the key is
    the same for both by construction -- this test survives key rotation, and
    needs no matrix, no column consistency and no prefix dimension.

    The digit run is described, not spelled, so a prompt-aware history cannot
    mask both lines. H must equal the token context h: a longer run repeats
    its own window inside line two and masks both types alike (the H=5 effect).

    A clean model is the check that the A/B asymmetry itself is not a signal.
    """
    print(f"\n### masking-differential test ({label})")
    words, example = forced_words(args) or WORD_LISTS[3]
    H, prefix = args.context, PREFIXES[0]
    log = response_log(args)
    fast = use_fast_path(ep, args)
    rng = np.random.default_rng(args.seed)
    print(f"  word list: {words}, H={H}, prefix={prefix!r}")

    # Build every (context, type) prompt first and query in one call: on the
    # batch transport each sample_texts call is a submission and a wait.
    cells = []
    for c in DIGITS:
        i = DIGITS.index(c)
        other = DIGITS[(i + 4) % len(DIGITS)]
        head = DIGITS[(i + 2) % len(DIGITS)]
        for typ in ("A", "B"):
            if args.masking_variant == "mixed":
                tail = c if typ == "A" else other
                prompt = build_two_line_mixed_prompt(prefix, str(head), str(tail),
                                                     str(c), H, words)
            else:
                prompt = build_two_line_prompt(prefix, str(c if typ == "A" else other),
                                               str(c), H, words)
            cells.append((c, typ, prompt))

    samples: dict = {c: [np.zeros(len(words)), np.zeros(len(words))] for c in DIGITS}
    raw: dict = {c: [] for c in DIGITS}
    if fast:
        k = max(1, args.k_gens)
        per = max(1, args.n_samples // k)
        flat = [(c, typ, prompt) for c, typ, prompt in cells for _ in range(k)]
        probes = ep.choice_probs([p for _, _, p in flat],
                                 [str(c) for c, _, _ in flat], H, words)
        for (c, typ, _), res in zip(flat, probes):
            if res is None:
                continue
            draws = rng.choice(len(words), size=per, p=res[0])
            samples[c][typ == "B"] += np.bincount(draws, minlength=len(words))
            raw[c] += [(typ, int(w)) for w in draws]
            if log:
                log.write(kind="probe", tag=tag, test="masking", c=c, type=typ,
                          probs=[float(x) for x in res[0]])
    else:
        texts = ep.sample_texts([p for _, _, p in cells], args.n_samples)
        for (c, typ, _), outs in zip(cells, texts):
            t2 = str(c) * H
            for text in outs:
                w = identify_word(text, words, t2)
                if log:
                    log.write(kind="response", tag=tag, test="masking", c=c,
                              type=typ, response=text,
                              parsed=None if w is None else words[w])
                if w is not None:
                    samples[c][typ == "B"][w] += 1
                    raw[c].append((typ, w))
    samples = {c: (a, b) for c, (a, b) in samples.items()}
    if args.verbose:
        for c in DIGITS:
            print(f"  c={c}: A={samples[c][0].astype(int)}  B={samples[c][1].astype(int)}")

    observed = masking_statistic(samples)
    hits = 0
    for _ in range(args.permutations):
        shuffled = {}
        for c, labelled in raw.items():
            labels = [t for t, _ in labelled]
            rng.shuffle(labels)
            a = np.zeros(len(words)); b = np.zeros(len(words))
            for t, (_, w) in zip(labels, labelled):
                (a if t == "A" else b)[w] += 1
            shuffled[c] = (a, b)
        hits += masking_statistic(shuffled) >= observed
    p = hits / args.permutations
    p_upper = 1.0 if hits == args.permutations else float(
        beta.ppf(0.995, hits + 1, args.permutations - hits))
    verdict = "MASKING FINGERPRINT DETECTED" if p_upper < 0.05 else "no evidence"
    print(f"  T={observed:.1f}  p={p:.4f}  p99={p_upper:.4f}  ->  {verdict}")
    return observed, p_upper


def use_fast_path(ep: Endpoint, args) -> bool:
    """Read scores when the endpoint offers them and the user did not opt out."""
    return ep.supports_logits and not args.slow


def build_endpoints(args) -> list[tuple[Endpoint, str, str]]:
    """(endpoint, label, cache tag) triples for this run."""
    if args.endpoint in ("anthropic", "anthropic-batch"):
        batch = args.endpoint == "anthropic-batch"
        cls = AnthropicBatchEndpoint if batch else AnthropicEndpoint

        # Budget first, credentials second: the cost preview is the thing worth
        # seeing before you go find a key, not after.
        cells = len(PREFIXES) * len(DIGITS)
        tag = "api_" + "".join(c if c.isalnum() else "-" for c in args.api_model)
        if args.t2_style != "digits":
            tag += f"_{args.t2_style}"
        if args.preamble:
            tag += "_pre"
        if args.probe != "echo":
            tag += f"_{args.probe}"
        if args.masking_test:
            # 9 contexts x 2 types, one prefix, no matrix.
            n_h, cells = 1, 2 * len(DIGITS)
        elif args.estimate_context:
            # Alg. 1 probes h_max run lengths x 9 perturbations x 9 digits.
            n_h = args.h_max
            cells = 9 * len(DIGITS)
        elif args.sweep_context:
            # Only price the H values that are not already cached on disk.
            hs = parse_h_range(args.sweep_context)
            if args.data_prefix:
                hs = [h for h in hs
                      if not os.path.exists(f"{args.data_prefix}_{tag}_H{h}.npz")]
            n_h = len(hs)
        else:
            n_h = 1
        probe = 0 if args.words else len(PREFIXES)
        planned = cells * args.n_samples * n_h + probe
        if batch:
            # The batch path over-requests so the caller's retry round is
            # usually unnecessary; price what it will actually send.
            planned = round(planned * args.oversample)
        print(f"endpoint: {args.api_model} (black box; no control endpoint exists)"
              f"{' [batch]' if batch else ''}")
        out_tokens = int(args.preamble_words * 1.5) + 25 if args.preamble else None
        print(f"budget:   {cls.estimate_cost(args.api_model, planned, out_tokens=out_tokens)}, "
              f"cap {args.max_requests}")
        if planned > args.max_requests:
            per_cell = max(1, args.max_requests // (cells * n_h))
            print(f"warning:  the plan needs {planned} requests but --max-requests "
                  f"is {args.max_requests}; later cells would be starved and fall "
                  f"back to smoothing priors. Raise the cap, or lower --n-samples "
                  f"to {per_cell}.")
        extra = {"oversample": args.oversample,
                 "poll_seconds": args.poll_seconds} if batch else {}
        system = PREAMBLE_SYSTEM if args.preamble else None
        ep = cls(
            model=args.api_model,
            temperature=args.temperature,
            max_tokens=args.api_max_tokens,
            system=system,
            concurrency=args.concurrency,
            nonce=not args.no_nonce,
            max_requests=args.max_requests,
            **extra,
        )
        if batch and args.retries > 1:
            print(f"note:     --retries is {args.retries}; on the batch transport "
                  f"each retry is another submission and another wait. "
                  f"--oversample {args.oversample} is there so the first round "
                  f"usually suffices.")
        if args.no_nonce:
            print("warning:  per-request nonce disabled; identical prompts may be "
                  "served from a provider-side cached prefix")
        # The cache tag carries the model: two models under one --data-prefix
        # must never read each other's matrices.
        return [(ep, f"{args.api_model} endpoint", tag)]

    from transformers import SynthIDTextWatermarkingConfig

    wm = SynthIDTextWatermarkingConfig(keys=WATERMARK_KEYS, ngram_len=NGRAM_LEN)
    common = dict(temperature=args.temperature, device=args.device,
                  batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    variants = [v.strip() for v in args.local_variants.split(",")]
    if args.skip_control:
        variants = [v for v in variants if v != "control"]
    base = LocalHFEndpoint(args.model, wm, **common)
    table = {
        "synthid": (base, "SynthID-Text endpoint", "wm"),
        # Same weights each time -- see LocalHFEndpoint.variant.
        "control": (base.variant(None), "unwatermarked control endpoint", "control"),
        "prompt-aware": (base.variant(wm, prompt_aware=True),
                         "SynthID-Text with prompt-aware masking", "wm-pa"),
    }
    return [table[v] for v in variants]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint",
                    choices=("local", "anthropic", "anthropic-batch"),
                    default="local")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="local HF model id")
    ap.add_argument("--api-model", default="claude-haiku-4-5",
                    help="Claude model id when --endpoint anthropic")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--context", type=int, default=NGRAM_LEN - 1,
                    help="H, the probed context size; must equal h for SynthID")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=10, help="local only")
    ap.add_argument("--max-new-tokens", type=int, default=28, help="local only")
    ap.add_argument("--concurrency", type=int, default=8, help="API only")
    ap.add_argument("--api-max-tokens", type=int, default=64, help="API only")
    ap.add_argument("--max-requests", type=int, default=5000,
                    help="hard cap on API calls; the run degrades rather than overspends")
    ap.add_argument("--oversample", type=float, default=1.3,
                    help="batch only: over-request this factor of samples per cell "
                         "so the retry round is usually unnecessary")
    ap.add_argument("--poll-seconds", type=float, default=15.0,
                    help="batch only: how often to poll a submitted batch")
    ap.add_argument("--no-nonce", action="store_true",
                    help="API only: stop prepending a per-request nonce")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--slow", action="store_true",
                    help="pure black box: parse sampled text instead of reading scores")
    ap.add_argument("--estimate-context", action="store_true",
                    help="Alg. 1 perturbation probe; expensive and +-1 noisy")
    ap.add_argument("--sweep-context", default=None, metavar="RANGE",
                    help="run the presence test at each H (e.g. '2-6') and read "
                         "h off the top of the detected band")
    ap.add_argument("--h-max", type=int, default=6)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t2-style", choices=T2_STYLES, default="digits",
                    help="'digits': one H-digit word; 'words': H digit words of "
                         "distinct lengths (covers word-level hashing, no "
                         "repeated n-gram windows)")
    ap.add_argument("--probe", choices=PROBES, default="echo",
                    help="'echo': the paper's prompt, context spelled out and "
                         "echoed; 'described': the model constructs the digit run "
                         "from a description, so it never appears in the prompt")
    ap.add_argument("--masking-test", action="store_true",
                    help="run the masking-differential test instead of the matrix")
    ap.add_argument("--masking-variant", choices=("same", "mixed"), default="same",
                    help="'same': line one repeats the whole number in type A; "
                         "'mixed': line one is head x3 + tail x(H-3) in both types, "
                         "so no type shows an identical number twice")
    ap.add_argument("--k-gens", type=int, default=5,
                    help="masking test, local fast path: generations per cell")
    ap.add_argument("--local-variants", default="synthid,control",
                    help="comma list of synthid, control, prompt-aware")
    ap.add_argument("--preamble-words", type=int, default=80,
                    help="length of the paragraph requested by --preamble")
    ap.add_argument("--preamble", action="store_true",
                    help="make the model write a paragraph before the probe "
                         "sentence (tests an output-length gate on the watermark)")
    ap.add_argument("--words", default=None, metavar="A,B,C,D",
                    help="skip the phase-1 probe and use this word list")
    ap.add_argument("--example", default="strawberries",
                    help="format-demo word for --words; must not be in the list")
    ap.add_argument("--data-prefix", default=None,
                    help="cache/reuse the collected probability matrices under this prefix")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    global PREAMBLE_WORDS
    PREAMBLE_WORDS = args.preamble_words
    if args.endpoint == "local":
        import torch
        torch.manual_seed(args.seed)

    for ep, label, tag in build_endpoints(args):
        if args.probe != "echo" and args.endpoint == "local":
            tag += f"_{args.probe}"
        if args.masking_test:
            masking_test(ep, args, label,
                         tag + ("_maskmix" if args.masking_variant == "mixed" else "_mask"))
        elif args.sweep_context:
            sweep_context(ep, args, label, tag)
        elif args.estimate_context:
            estimate_context_size(ep, args, label, tag)
        else:
            run_test(ep, args, label, tag)
        n_req = getattr(ep, "n_requests", None)
        if n_req is not None:
            print(f"  API calls: {n_req} sent, {ep.n_failed} failed "
                  f"({getattr(ep, 'n_refused', 0)} of them refusals)")


if __name__ == "__main__":
    main()
