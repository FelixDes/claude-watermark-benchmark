"""Swappable sampling endpoints for the Red-Green watermark presence test.

The test only ever needs one thing from a model: *sample a completion, many
times, independently*. That is the contract below. A local `transformers` model
can additionally hand over the score vector at the choice position, which makes
data collection ~100x cheaper, so the contract exposes that as an optional
capability rather than a requirement.

    Endpoint
      .name              str, for logging
      .supports_logits   bool, whether choice_probs() works
      .sample_texts()    required -- n independent completions per prompt
      .choice_probs()    optional -- exact choice distribution per prompt

Three implementations:

    LocalHFEndpoint         transformers on CUDA, optionally SynthID-watermarked.
                            Ground truth: we control whether a watermark is there.
    AnthropicEndpoint       a Claude model over the Messages API, one request per
                            sample. Text only, so the strict black-box path is
                            the only one available. Results stream back, which is
                            what you want while exploring.
    AnthropicBatchEndpoint  the same model over the Batches API: half the price,
                            separate and much higher rate limits, at the cost of
                            waiting for the whole submission. Same distribution --
                            each batch item is an independent inference with the
                            same parameters, and the statistic is a count of
                            parsed words that cannot tell how the text arrived.

Running against a hosted model needs three things the local path gets for free.

*Sampling must stay on.* The test reads a *distribution* over four words; a
greedy endpoint returns the same word every time, every cell of the matrix
collapses to the same logit, and the statistic is identically zero. Claude
Haiku 4.5 accepts `temperature`, so we set it explicitly -- but the SDK has
since dropped `temperature` from its typed signature, because the current model
generation (Opus 5, Sonnet 5, Fable 5, the 4.6+ family) removed sampling
parameters and 400s on them. So the parameter is routed through `extra_body`
whenever the installed signature does not name it, and a 400 that names it
makes us drop it and fall back to the provider's default sampling -- still
stochastic, but no longer under our control, which we say loudly.

*Thinking should be off.* It burns tokens for a one-word answer, and if the
model rehearses the sentence while reasoning, SynthID's repeated-context masking
can switch the watermark off at the answer -- a false negative. Models that
think by default get `thinking: {"type": "disabled"}`; older ones simply omit
the field. Fable/Mythos cannot disable it at all: they run at effort "low" with
a warning that a negative result is weaker there. The watermark's context for
the chosen word is the echoed sentence in the *visible* answer, so thinking
cannot manufacture a false positive.

*Provider caching must not collapse the samples.* Anthropic prompt caching is
opt-in, so simply never passing `cache_control` is most of the answer. For the
rest we prepend a per-request nonce, which both defeats any prefix reuse and
guarantees the N requests in one cell are not served from one cached prefix.
The nonce goes at the very front of the system prompt, as far as possible from
the choice position: the watermark context is the handful of tokens immediately
preceding the chosen word, deep inside the model's own echoed sentence, so a
leading nonce cannot enter it. Placing it at the end of the user turn would.
"""

from __future__ import annotations

import inspect
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, Sequence

import numpy as np


class Endpoint(Protocol):
    """Sampling-only view of a chat model."""

    name: str
    supports_logits: bool

    def sample_texts(self, prompts: Sequence[str], n_samples: int) -> list[list[str]]:
        """Return `n_samples` independent completions for each prompt."""
        ...

    def choice_probs(self, prompts: Sequence[str], digits: Sequence[str], H: int,
                     words: Sequence[str]) -> list[tuple[np.ndarray, float] | None]:
        """Exact distribution over `words` at the choice position, per prompt.

        Returns (probs, mass) per prompt, where `mass` is how much of the full
        vocabulary distribution the word set accounts for -- a sanity check that
        the model really was about to answer. None where the probe failed.
        Only meaningful when `supports_logits`.
        """
        ...


# --------------------------------------------------------------------------
# Local transformers backend
# --------------------------------------------------------------------------


class LocalHFEndpoint:
    """A local causal LM, optionally watermarked with SynthID-Text.

    This is the ground-truth backend: `watermarking_config=None` gives a control
    endpoint that is known-clean, which is how the test's false-positive rate
    gets checked at all.
    """

    supports_logits = True

    def __init__(self, model_id: str, watermarking_config=None, temperature: float = 1.0,
                 device: str = "cuda", batch_size: int = 6, max_new_tokens: int = 28,
                 _shared=None, prompt_aware: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        flavour = (" (control)" if not watermarking_config
                   else "+synthid-prompt-aware" if prompt_aware else "+synthid")
        self.name = f"{model_id}{flavour}"
        self.prompt_aware = prompt_aware
        if _shared is not None:
            self.model, self.tokenizer = _shared
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=device
            ).eval()
        self.model_id = model_id
        self.watermarking_config = watermarking_config
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def variant(self, watermarking_config, prompt_aware: bool = False) -> "LocalHFEndpoint":
        """A second endpoint over the *same* loaded weights.

        The watermark is a generation-time logits processor, not a property of
        the checkpoint, so the watermarked endpoint, its prompt-aware variant
        and the control are all the same model. Loading it three times would
        just triple the VRAM.
        """
        return LocalHFEndpoint(
            self.model_id, watermarking_config, self.temperature, self.device,
            self.batch_size, self.max_new_tokens,
            _shared=(self.model, self.tokenizer), prompt_aware=prompt_aware,
        )

    def _generate(self, prompts: Sequence[str], want_scores: bool):
        torch = self._torch
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        extra = {}
        if self.watermarking_config is not None and self.prompt_aware:
            # Stateful per batch, so a fresh instance every call. Passed as a
            # plain logits processor instead of `watermarking_config`, which
            # would build the stock (prompt-blind) one.
            from transformers import LogitsProcessorList
            from watermarks import PromptAwareSynthIDProcessor
            proc = PromptAwareSynthIDProcessor(
                **self.watermarking_config.to_dict(), device=self.device)
            extra["logits_processor"] = LogitsProcessorList([proc])
        else:
            extra["watermarking_config"] = self.watermarking_config
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.temperature,
                top_k=0,
                top_p=1.0,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=want_scores,
                **extra,
            )
        gen = out.sequences[:, enc["input_ids"].shape[1]:].cpu()
        # Off the GPU immediately: [batch, steps, vocab] does not fit alongside
        # the watermark processor's own [batch, vocab, depth] buffers on 12 GB.
        scores = None
        if want_scores:
            scores = torch.stack(out.scores, dim=1).to("cpu", torch.float32)
        del out
        return gen, scores

    def sample_texts(self, prompts: Sequence[str], n_samples: int) -> list[list[str]]:
        flat = [p for p in prompts for _ in range(n_samples)]
        texts: list[str] = []
        for start in range(0, len(flat), self.batch_size):
            gen, _ = self._generate(flat[start:start + self.batch_size], want_scores=False)
            texts.extend(
                self.tokenizer.decode(g, skip_special_tokens=True) for g in gen
            )
        return [texts[i * n_samples:(i + 1) * n_samples] for i in range(len(prompts))]

    def choice_probs(self, prompts, digits, H, words):
        word_ids = [self.tokenizer.encode(" " + w, add_special_tokens=False)[0]
                    for w in words]
        if len(set(word_ids)) != len(word_ids):
            raise ValueError(f"word list {list(words)} lacks distinct first tokens")
        digit_ids = {self.tokenizer.encode(str(i), add_special_tokens=False)[0]
                     for i in range(10)}

        results: list[tuple[np.ndarray, float] | None] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = slice(start, start + self.batch_size)
            gen, scores = self._generate(prompts[chunk], want_scores=True)
            for j in range(gen.shape[0]):
                results.append(
                    self._read_choice(gen[j], scores[j], digit_ids, word_ids)
                )
        return results

    def _read_choice(self, gen_ids, scores, digit_ids: set[int], word_ids: list[int]):
        """Distribution at the token right after the *last* run of digits.

        The last run rather than a run of exactly H: a model asked to write a
        digit twelve times may write eleven, and a two-line probe has two runs
        with the second one being the position of interest. Either way the
        context before the choice is still a function of the digit alone.

        For SynthID this is exact rather than an approximation: its logits
        processor returns the log-probs of the already-watermarked distribution,
        so softmax over the score vector *is* the distribution the watermark
        produced.
        """
        ids = gen_ids.tolist()
        end = None
        run = 0
        for pos, tok in enumerate(ids):
            if tok in digit_ids:
                run += 1
            else:
                if run >= 2:
                    end = pos  # first non-digit after a run: the choice slot
                run = 0
        if end is None or end >= len(ids):
            return None
        probs = self._torch.softmax(scores[end].float(), dim=-1)
        sel = probs[word_ids]
        mass = float(sel.sum())
        if mass < 1e-4:
            return None  # model was not about to answer with the word set
        return (sel / sel.sum()).cpu().numpy(), mass


# --------------------------------------------------------------------------
# Anthropic API backend
# --------------------------------------------------------------------------

# Thinking is on by default on these; it has to be turned off explicitly.
_THINKS_BY_DEFAULT = ("claude-opus-5", "claude-sonnet-5")
# These cannot turn thinking off at all. Still testable -- the watermark's
# context for the chosen word is the echoed sentence in the *visible* answer,
# which thinking does not touch -- but with one caveat: if the model rehearses
# the sentence while thinking, SynthID's repeated-context masking can switch
# the watermark off at the answer. That yields false negatives, never false
# positives, so we warn, pin effort to its minimum, and proceed.
_THINKING_ALWAYS_ON = ("claude-fable-5", "claude-mythos-5")
# The current generation removed sampling parameters entirely (400 on
# `temperature`); skipping them up front saves a wasted round trip.
_NO_SAMPLING_PARAMS = ("claude-fable-5", "claude-mythos-5", "claude-opus-5",
                       "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7")

_SYSTEM = (
    "You complete sentences exactly as instructed. Reply with the completed "
    "sentence and nothing else: no preamble, no explanation, no quotation marks."
)


class _AnthropicBase:
    """Shared plumbing for the synchronous and batch Claude endpoints.

    No logits are available either way, so `supports_logits` is False and the
    caller is forced onto the text-parsing path -- which is the setting the paper
    actually describes.
    """

    supports_logits = False

    def __init__(self, model: str = "claude-haiku-4-5", temperature: float | None = 1.0,
                 max_tokens: int = 64, concurrency: int = 8, nonce: bool = True,
                 max_requests: int = 5000, client=None, system: str | None = None):
        import anthropic

        self.system_text = system or _SYSTEM

        self.always_thinks = any(model.startswith(m) for m in _THINKING_ALWAYS_ON)
        if self.always_thinks:
            print(f"  note: {model} cannot disable thinking. Effort pinned to "
                  f"'low'. A negative here is weaker than usual: if the model "
                  f"rehearses the sentence while thinking, repeated-context "
                  f"masking can hide the watermark at the answer.")
            if max_tokens < 128:
                # max_tokens covers thinking + answer; the 64-token short-answer
                # default would cut the answer off before it starts. An explicit
                # larger cap is left alone -- it may be there to bound cost.
                max_tokens = 2048

        self._anthropic = anthropic
        self.name = model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.nonce = nonce
        self.max_requests = max_requests
        self.client = client or anthropic.Anthropic(max_retries=6)
        if client is None and not self.client.auth_headers:
            raise RuntimeError(
                "no Anthropic credentials found. Export ANTHROPIC_API_KEY, or run "
                "`ant auth login` to store a profile the SDK picks up automatically."
            )

        # Params we would like to send but will drop if the model rejects them.
        self._optional = {}
        no_sampling = any(model.startswith(m) for m in _NO_SAMPLING_PARAMS)
        if temperature is not None and not no_sampling:
            self._optional["temperature"] = temperature
        elif temperature is not None:
            print(f"  note: {model} accepts no sampling parameters; using the "
                  f"provider default (stochastic, but not ours to set).")
        if any(model.startswith(m) for m in _THINKS_BY_DEFAULT):
            self._optional["thinking"] = {"type": "disabled"}
        if self.always_thinks:
            self._optional["output_config"] = {"effort": "low"}

        self._lock = threading.Lock()
        self.n_requests = 0
        self.n_failed = 0
        self.n_refused = 0
        self._consecutive_failures = 0

    def _request_params(self, prompt: str, params: dict) -> dict:
        """A single Messages request body, shared by both transports."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self._system(),
            "messages": [{"role": "user", "content": prompt}],
            **params,
        }

    # -- request plumbing ---------------------------------------------------

    def _system(self) -> str:
        if not self.nonce:
            return self.system_text
        # Leading position is deliberate: it breaks any provider-side prefix
        # reuse, and is far enough from the choice position that it cannot enter
        # the watermark's context window.
        return f"[request-id: {secrets.token_hex(8)}]\n{self.system_text}"

    def _drop_rejected_param(self, params: dict, message: str) -> bool:
        """Drop a parameter the model rejected, once, and report it."""
        low = message.lower()
        for key, hints in (("temperature", ("temperature", "top_p", "top_k")),
                           ("thinking", ("thinking",))):
            if key in params and any(h in low for h in hints):
                del params[key]
                with self._lock:
                    self._optional.pop(key, None)
                note = (
                    "  note: this model rejects `temperature`; sampling now uses the "
                    "provider default. The test needs a stochastic endpoint -- if it "
                    "turns out to be greedy, every cell collapses and S is 0."
                    if key == "temperature" else
                    "  note: this model rejects `thinking: disabled`; leaving it unset."
                )
                print(note)
                return True
        return False

    def choice_probs(self, prompts, digits, H, words):
        raise NotImplementedError(
            "the API returns text only; run the test with --slow"
        )

    #: Batch processing is billed at half the synchronous rate.
    cost_multiplier = 1.0

    @classmethod
    def estimate_cost(cls, model: str, n_requests: int, prompt_chars: int = 400,
                      out_tokens: int | None = None) -> str:
        """Rough per-run cost, from the published per-MTok rates.

        A classmethod so a run can be priced before credentials are resolved --
        the number you want before going to find a key, not after -- and so the
        batch subclass can apply its discount.
        """
        rates = {  # (input $/MTok, output $/MTok)
            "claude-haiku-4-5": (1.0, 5.0),
            "claude-sonnet-5": (3.0, 15.0),
            "claude-opus-5": (5.0, 25.0),
            "claude-fable-5": (10.0, 50.0),
        }
        rate = next((v for k, v in rates.items() if model.startswith(k)), None)
        if rate is None:
            return f"{n_requests} requests (no cached rate for {model})"
        tok_in = prompt_chars / 3.5 + 40  # prompt + system, chars-per-token approx
        thinks = any(model.startswith(m) for m in _THINKING_ALWAYS_ON)
        # Measured on Fable 5.1 at effort "low": output_tokens == the answer,
        # no thinking at all. Keep a small margin for models that do think.
        tok_out = out_tokens or (40 if thinks else 25)
        cost = n_requests * (tok_in * rate[0] + tok_out * rate[1]) / 1e6 * cls.cost_multiplier
        tail = " (batch, 50% off)" if cls.cost_multiplier != 1.0 else " list prices"
        return f"{n_requests} requests, ~${cost:.2f} at {model}{tail}"


class AnthropicEndpoint(_AnthropicBase):
    """Synchronous Messages API: one HTTP request per sample, run on a pool.

    Results stream back immediately, which is what you want while exploring --
    the sweep prints a row per H as it goes. For full-fidelity runs the batch
    endpoint below is half the price and not rate limited.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The SDK drops parameters from its typed signature once the current
        # model generation stops accepting them -- `temperature` is gone as of
        # 1.2.0 -- but older models still honour them on the wire. Route
        # anything the installed signature does not name through `extra_body`
        # so this keeps working across SDK versions in both directions. The
        # batch transport needs none of this: its per-item `params` is a plain
        # dict that serialises through untouched.
        self._named = set(
            inspect.signature(self.client.messages.create).parameters
        )

    def _split_params(self, params: dict) -> dict:
        named = {k: v for k, v in params.items() if k in self._named}
        extra = {k: v for k, v in params.items() if k not in self._named}
        if extra:
            named["extra_body"] = extra
        return named

    def _one(self, prompt: str) -> str | None:
        with self._lock:
            if self.n_requests >= self.max_requests:
                return None
            self.n_requests += 1

        params = dict(self._optional)
        while True:
            try:
                resp = self.client.messages.create(
                    **self._split_params(self._request_params(prompt, params))
                )
                break
            except self._anthropic.BadRequestError as exc:
                dropped = self._drop_rejected_param(params, str(exc))
                if not dropped:
                    raise
            except (self._anthropic.AuthenticationError,
                    self._anthropic.PermissionDeniedError,
                    self._anthropic.NotFoundError):
                # Nothing about these gets better by sending 2000 more requests.
                raise
            except self._anthropic.APIStatusError:
                # Transient (overloaded, 5xx); the SDK already retried. Losing a
                # few samples out of a cell is fine -- the estimator is a count.
                with self._lock:
                    self.n_failed += 1
                    self._consecutive_failures += 1
                    dead = self._consecutive_failures >= 25
                if dead:
                    raise RuntimeError(
                        "25 consecutive API failures; aborting rather than "
                        "collecting a matrix of smoothing priors"
                    )
                return None

        with self._lock:
            self._consecutive_failures = 0
        if getattr(resp, "stop_reason", None) == "refusal":
            with self._lock:
                self.n_failed += 1
                self.n_refused += 1
            return None
        return "".join(b.text for b in resp.content if b.type == "text")

    def sample_texts(self, prompts: Sequence[str], n_samples: int) -> list[list[str]]:
        jobs = [(i, p) for i, p in enumerate(prompts) for _ in range(n_samples)]
        out: list[list[str]] = [[] for _ in prompts]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for (i, _), text in zip(jobs, pool.map(lambda j: self._one(j[1]), jobs)):
                if text is not None:
                    out[i].append(text)
        return out


class AnthropicBatchEndpoint(_AnthropicBase):
    """Messages Batches API: one submission per call, then wait.

    Same distribution, half the price. Each batch item is an independent
    inference with the same parameters, so nothing about asynchronous processing
    touches the choice distribution the test measures -- the statistic is a count
    of parsed words per cell and does not care how the text arrived.

    Two things do change, and both are handled here rather than by the caller:

    *Retries cost a whole round trip.* The caller's retry loop re-queries cells
    that came back under-sampled; on this transport each retry is another batch
    and another wait. So we over-request by `oversample` up front and return
    everything that lands, which makes the second round usually unnecessary.
    Uneven sample counts across cells are fine -- the estimator is a proportion,
    and parse failures already made them uneven on any transport.

    *Results come back in arbitrary order.* They are keyed by `custom_id`, never
    by position.

    Note this is one batch per `sample_texts` call, i.e. per H in a sweep, not
    one batch for the whole sweep. Folding every H into a single submission
    would mean restructuring the caller's collect/score loop; the win from that
    is one wait instead of five, not a cheaper or better estimate.
    """

    cost_multiplier = 0.5

    #: API ceiling is 100k requests / 256MB per batch; stay well under both.
    MAX_BATCH = 50_000

    def __init__(self, *args, oversample: float = 1.3, poll_seconds: float = 15.0,
                 timeout_seconds: float = 24 * 3600, **kwargs):
        super().__init__(*args, **kwargs)
        self.oversample = oversample
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def sample_texts(self, prompts: Sequence[str], n_samples: int) -> list[list[str]]:
        per_prompt = max(1, round(n_samples * self.oversample))
        jobs = [(i, p) for i, p in enumerate(prompts) for _ in range(per_prompt)]

        with self._lock:
            room = max(0, self.max_requests - self.n_requests)
        if len(jobs) > room:
            jobs = jobs[:room]
            print(f"  note: request cap reached; submitting {len(jobs)} of "
                  f"{len(prompts) * per_prompt} planned samples")
        if not jobs:
            return [[] for _ in prompts]
        with self._lock:
            self.n_requests += len(jobs)

        out: list[list[str]] = [[] for _ in prompts]
        params = dict(self._optional)
        for start in range(0, len(jobs), self.MAX_BATCH):
            chunk = jobs[start:start + self.MAX_BATCH]
            for i, text in self._run_batch(chunk, params):
                out[i].append(text)
        return out

    # -- batch plumbing -----------------------------------------------------

    def _run_batch(self, jobs, params: dict):
        """Submit, wait, and yield (prompt_index, text) for everything that
        succeeded. Retries once if the model rejects an optional parameter."""
        while True:
            requests = [
                {"custom_id": f"j{k}",
                 "params": self._request_params(prompt, params)}
                for k, (_, prompt) in enumerate(jobs)
            ]
            batch = self.client.messages.batches.create(requests=requests)
            print(f"  batch {batch.id}: {len(requests)} requests submitted")
            batch = self._await(batch.id)

            results = list(self.client.messages.batches.results(batch.id))
            complaint = self._first_param_complaint(results, params)
            if complaint and self._drop_rejected_param(params, complaint):
                print("  resubmitting the batch without it")
                continue
            break

        succeeded = 0
        for item in results:
            k = int(item.custom_id[1:])
            if item.result.type != "succeeded":
                with self._lock:
                    self.n_failed += 1
                continue
            message = item.result.message
            if getattr(message, "stop_reason", None) == "refusal":
                with self._lock:
                    self.n_failed += 1
                    self.n_refused += 1
                continue
            succeeded += 1
            text = "".join(b.text for b in message.content if b.type == "text")
            yield jobs[k][0], text
        print(f"  batch {batch.id}: {succeeded}/{len(jobs)} succeeded")

    def _await(self, batch_id: str):
        """Poll until the batch ends, reporting progress as it goes."""
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return batch
            if time.monotonic() > deadline:
                # The batch keeps running server-side; only this wait gives up.
                # Nothing here resumes it, so hand over the id for manual pickup.
                raise RuntimeError(
                    f"batch {batch_id} still {batch.processing_status} after "
                    f"{self.timeout_seconds:.0f}s. It is still being processed and "
                    f"billed; fetch it yourself with "
                    f"client.messages.batches.results('{batch_id}') once it ends."
                )
            c = batch.request_counts
            print(f"    {batch.processing_status}: {c.succeeded} done, "
                  f"{c.processing} processing, {c.errored} errored",
                  flush=True)
            time.sleep(self.poll_seconds)

    def _first_param_complaint(self, results, params: dict) -> str | None:
        """An invalid-request message naming a parameter we chose to send.

        Per-item errors replace the synchronous path's 400, so a model that
        rejects `temperature` or `thinking` shows up here instead.
        """
        for item in results:
            if item.result.type != "errored":
                continue
            # MessageBatchErroredResult.error is an ErrorResponse whose own
            # .error carries the typed error with the message.
            err = getattr(item.result.error, "error", item.result.error)
            message = str(getattr(err, "message", err))
            low = message.lower()
            if any(k in low for k in params):
                return message
        return None


# --------------------------------------------------------------------------
# Google Gemini backend
# --------------------------------------------------------------------------


class GeminiEndpoint:
    """A Gemini model over the Google GenAI API, sampled as a strict black box.

    The positive-control candidate: Google is SynthID-Text's author and ships
    it in the Gemini app; whether the API path carries it is what a run here
    finds out. Same contract, same three concerns as the Anthropic backends:
    sampling on (`temperature` is honoured here), thinking off
    (`thinking_budget=0`; models that cannot go to zero keep their minimum and
    say so), and a per-request nonce at the head of the system instruction.
    """

    supports_logits = False
    cost_multiplier = 1.0

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float | None = 1.0,
                 max_tokens: int = 64, concurrency: int = 8, nonce: bool = True,
                 max_requests: int = 5000, client=None, system: str | None = None):
        import os
        from google import genai
        from google.genai import types

        self._types = types
        self.name = model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.nonce = nonce
        self.max_requests = max_requests
        self.system_text = system or _SYSTEM
        if client is None:
            key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("no Gemini credentials: export GOOGLE_API_KEY (or GEMINI_API_KEY)")
            client = genai.Client(api_key=key)
        self.client = client
        self._thinking_off = True
        self._lock = threading.Lock()
        self.n_requests = 0
        self.n_failed = 0
        self.n_refused = 0
        self._consecutive_failures = 0

    def _system(self) -> str:
        if not self.nonce:
            return self.system_text
        return f"[request-id: {secrets.token_hex(8)}]\n{self.system_text}"

    def _config(self):
        t = self._types
        cfg = dict(system_instruction=self._system(), max_output_tokens=self.max_tokens,
                   candidate_count=1)
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        if self._thinking_off:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=0)
        return t.GenerateContentConfig(**cfg)

    def _one(self, prompt: str) -> str | None:
        with self._lock:
            if self.n_requests >= self.max_requests:
                return None
            self.n_requests += 1
        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config())
                break
            except Exception as exc:  # google.genai.errors.APIError and friends
                msg = str(exc).lower()
                if self._thinking_off and "thinking" in msg:
                    # Model cannot go to zero thinking; keep its minimum.
                    self._thinking_off = False
                    print("  note: this model rejects thinking_budget=0; leaving thinking at its minimum")
                    continue
                if any(k in msg for k in ("api key", "permission", "not found", "invalid argument")):
                    raise
                time.sleep(min(2 ** attempt, 30))  # 429 / 5xx
        else:
            with self._lock:
                self.n_failed += 1
                self._consecutive_failures += 1
                if self._consecutive_failures >= 25:
                    raise RuntimeError("25 consecutive Gemini failures; aborting")
            return None
        with self._lock:
            self._consecutive_failures = 0
        cands = getattr(resp, "candidates", None) or []
        finish = str(getattr(cands[0], "finish_reason", "")) if cands else "BLOCKED"
        text = getattr(resp, "text", None)
        if not text or "SAFETY" in finish or "BLOCK" in finish:
            with self._lock:
                self.n_failed += 1
                self.n_refused += 1
            return None
        return text

    def sample_texts(self, prompts: Sequence[str], n_samples: int) -> list[list[str]]:
        jobs = [(i, p) for i, p in enumerate(prompts) for _ in range(n_samples)]
        out: list[list[str]] = [[] for _ in prompts]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for (i, _), text in zip(jobs, pool.map(lambda j: self._one(j[1]), jobs)):
                if text is not None:
                    out[i].append(text)
        return out

    def choice_probs(self, prompts, digits, H, words):
        raise NotImplementedError("the API returns text only; run the test with --slow")

    @classmethod
    def estimate_cost(cls, model: str, n_requests: int, prompt_chars: int = 400,
                      out_tokens: int | None = None) -> str:
        rates = {"gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50),
                 "gemini-2.5-pro": (1.25, 10.0)}
        rate = next((v for k, v in rates.items() if model.startswith(k)), None)
        if rate is None:
            return f"{n_requests} requests (no cached rate for {model})"
        tok_in = prompt_chars / 3.5 + 40
        cost = n_requests * (tok_in * rate[0] + (out_tokens or 25) * rate[1]) / 1e6
        return f"{n_requests} requests, ~${cost:.2f} at {model} list prices"
