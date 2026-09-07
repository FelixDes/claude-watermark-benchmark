"""Offline checks for the Red-Green test and its API transports.

No credentials, no GPU. Fake clients stand in for the Anthropic SDK and behave
like a chat model that echoes the sentence and picks a fruit -- with or without
a context-keyed bias, i.e. with or without a Red-Green watermark. The point is
to exercise the whole pipeline (prompting, parsing, matrix, statistic,
permutation test, sweep, logging) against known ground truth before a single
paid request goes out.

    python test_blackbox.py          # plain
    pytest test_blackbox.py          # also works

The fakes are deliberately unkind: batch results come back shuffled, a few
percent of items error, and the batch fake asserts that `temperature` and the
nonce actually reached the per-item params.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import tempfile
import types

import numpy as np

import blackbox_redgreen as bb
from endpoints import AnthropicBatchEndpoint, AnthropicEndpoint

WORDS = bb.WORD_LISTS[0][0]


# --------------------------------------------------------------------------
# A fake model
# --------------------------------------------------------------------------


def answer(prompt: str, words, h_true: int | None) -> str:
    """Echo the sentence and pick a fruit.

    `h_true=None` is an unwatermarked model: a mild per-prefix preference and
    nothing else. Otherwise the bias is keyed on the digit run exactly when its
    length equals `h_true`, mimicking SynthID: shorter runs let the context
    bleed into the varying prefix, longer ones trip repeated-context masking.
    """
    m = re.search(r'"([^"]*?) (\d+)"', prompt)
    prefix, digits = m.group(1), m.group(2)
    base = np.array([1.2, 0.6, 0.9, 0.4])
    base = base + 0.25 * np.array([hash((prefix, w)) % 7 for w in words]) / 7
    if h_true is not None:
        if len(digits) == h_true:
            key = digits
        elif len(digits) < h_true:
            key = f"{prefix}|{digits}"
        else:
            key = None
        if key is not None:
            seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            base = base + np.random.default_rng(seed).choice([-1.4, 1.4], size=4)
    p = np.exp(base) / np.exp(base).sum()
    return f"{prefix} {digits} {words[np.random.choice(4, p=p)]}."


def text_block(text):
    return types.SimpleNamespace(type="text", text=text)


class FakeSyncClient:
    def __init__(self, h_true):
        self.h_true = h_true
        self.calls = 0
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, model, max_tokens, system, messages, **kw):
        self.calls += 1
        assert system.startswith("[request-id:"), "nonce missing from system prompt"
        assert kw.get("extra_body", {}).get("temperature") == 1.0, \
            "temperature must reach the wire via extra_body"
        return types.SimpleNamespace(
            content=[text_block(answer(messages[0]["content"], WORDS, self.h_true))],
            stop_reason="end_turn",
        )


class FakeBatchClient:
    def __init__(self, h_true, fail_rate=0.03):
        self.h_true = h_true
        self.fail_rate = fail_rate
        self.submitted = 0
        self.batches = 0
        self._store = {}
        self.messages = types.SimpleNamespace(
            batches=types.SimpleNamespace(
                create=self._create, retrieve=self._retrieve, results=self._results,
            )
        )

    def _create(self, requests):
        bid = f"batch_{self.batches}"
        self._store[bid] = list(requests)
        self.submitted += len(self._store[bid])
        self.batches += 1
        return types.SimpleNamespace(id=bid)

    def _retrieve(self, bid):
        n = len(self._store[bid])
        return types.SimpleNamespace(
            id=bid, processing_status="ended",
            request_counts=types.SimpleNamespace(
                succeeded=n, processing=0, errored=0, canceled=0, expired=0),
        )

    def _results(self, bid):
        items = []
        for req in self._store[bid]:
            p = req["params"]
            assert p.get("temperature") == 1.0, "temperature missing from batch params"
            assert p["system"].startswith("[request-id:"), "nonce missing"
            if random.random() < self.fail_rate:
                res = types.SimpleNamespace(type="errored", error=None)
            else:
                res = types.SimpleNamespace(
                    type="succeeded",
                    message=types.SimpleNamespace(
                        stop_reason="end_turn",
                        content=[text_block(answer(p["messages"][0]["content"],
                                                   WORDS, self.h_true))],
                    ),
                )
            items.append(types.SimpleNamespace(custom_id=req["custom_id"], result=res))
        random.shuffle(items)  # arbitrary order is part of the contract
        return items


def make_args(**over):
    base = dict(context=4, n_samples=25, permutations=1500, retries=2,
                verbose=False, seed=0, slow=True, data_prefix=None,
                sweep_context=None, words=None, example="strawberries",
                t2_style="digits", preamble=False, probe="echo", k_gens=5,
                masking_variant="same")
    base.update(over)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_sync_detects_watermark_and_clears_control():
    random.seed(0); np.random.seed(0)
    for h_true, expect_hit in ((4, True), (None, False)):
        fake = FakeSyncClient(h_true)
        ep = AnthropicEndpoint(model="claude-haiku-4-5", client=fake,
                               concurrency=4, max_requests=10**6)
        _, verdicts, _ = bb.run_test(ep, make_args(), "sync", "t", quiet=True)
        p99 = verdicts["colstd"][2]
        assert (p99 < 0.05) == expect_hit, (h_true, verdicts)
        assert ep.n_requests == fake.calls


def test_batch_detects_watermark_with_shuffled_partial_results():
    random.seed(0); np.random.seed(0)
    fake = FakeBatchClient(h_true=4)
    ep = AnthropicBatchEndpoint(model="claude-haiku-4-5", client=fake,
                                max_requests=10**6, oversample=1.3, poll_seconds=0)
    _, verdicts, _ = bb.run_test(ep, make_args(), "batch", "t", quiet=True)
    assert verdicts["colstd"][2] < 0.05, verdicts
    # word-list probe + one batch per collect round
    assert fake.batches >= 2
    # oversampling: 90 cells x round(25 * 1.3)
    assert fake.submitted >= 90 * 32
    assert 0 < ep.n_failed < 0.1 * fake.submitted


def test_sweep_recovers_context_size():
    random.seed(0); np.random.seed(0)
    fake = FakeSyncClient(h_true=4)
    ep = AnthropicEndpoint(model="claude-haiku-4-5", client=fake,
                           concurrency=8, max_requests=10**6)
    with tempfile.TemporaryDirectory() as d:
        args = make_args(sweep_context="2-6", data_prefix=f"{d}/sweep")
        detected = bb.sweep_context(ep, args, "sweep", "t")
        assert max(detected) == 4, detected
        # raw responses were logged as they arrived
        n_lines = sum(1 for _ in open(f"{d}/sweep_responses.jsonl"))
        assert n_lines == 5 * 90 * 25, n_lines


def test_param_routing_per_model():
    class Probe:
        """Declares the real SDK's named parameters, so the endpoint's
        signature introspection routes exactly as it would in production:
        `thinking` is named, `temperature` is not."""
        def __init__(self): self.seen = {}
        def create(self, *, model, max_tokens, messages, system=None, thinking=None,
                   extra_body=None, **kw):
            self.seen = dict(model=model, thinking=thinking, extra_body=extra_body, **kw)
            return types.SimpleNamespace(content=[text_block("x")], stop_reason="end_turn")
    # (model, thinking param, extra_body) -- Haiku takes temperature on the
    # wire; Opus 5 rejects all sampling params, so none are sent.
    for model, want_thinking, want_extra in (
        ("claude-haiku-4-5", None, {"temperature": 1.0}),
        ("claude-opus-5", {"type": "disabled"}, None),
    ):
        probe = Probe()
        client = types.SimpleNamespace(messages=probe)
        AnthropicEndpoint(model=model, client=client)._one("hi")
        assert probe.seen.get("thinking") == want_thinking, (model, probe.seen)
        assert probe.seen["extra_body"] == want_extra, (model, probe.seen)


def test_batch_param_complaint_is_read_from_nested_error():
    from anthropic.types.messages import MessageBatchErroredResult
    from anthropic.types.shared import ErrorResponse, InvalidRequestError
    item = types.SimpleNamespace(custom_id="j0", result=MessageBatchErroredResult(
        type="errored",
        error=ErrorResponse(type="error", request_id=None,
                            error=InvalidRequestError(
                                type="invalid_request_error",
                                message="temperature: Extra inputs are not permitted")),
    ))
    ep = AnthropicBatchEndpoint.__new__(AnthropicBatchEndpoint)
    assert ep._first_param_complaint([item], {"temperature": 1.0})
    assert ep._first_param_complaint([item], {"thinking": {}}) is None


def test_parser_reads_word_after_echoed_t2():
    words = ["apples", "bananas", "oranges", "pears"]
    t2 = "3 33 333"
    # preamble mentions a candidate; the echo decides
    assert bb.identify_word("I like pears. I ate 3 33 333 oranges.", words, t2) == 2
    # no echo -> reference rule still applies
    assert bb.identify_word("I ate oranges.", words, t2) == 2
    assert bb.identify_word("oranges and pears", words, t2) is None
    assert bb.t2_string(3, 4, "words") == "3 33 333 3333"
    assert bb.t2_string(3, 4) == "3333"


def test_described_prompt_carries_no_digit_run():
    words = ["apples", "bananas", "oranges", "pears"]
    p = bb.build_prompt("I ate", "3333", words, "strawberries", probe="described")
    assert "3333" not in p and "four copies" in p and "<number>" in p
    # the digit itself appears once, as the description ("the digit 3,")
    assert p.count("digit 3") == 1
    two = bb.build_two_line_prompt("I ate", "4", "3", 4, words)
    assert "4444" not in two and "3333" not in two
    # the parser still anchors on the run the model produces
    assert bb.identify_word("I ate 3333 pears.", words, "3333") == 3
    assert bb.identify_word("I ate 4444 apples.\nI ate 3333 oranges.", words, "3333") == 2


def test_masking_statistic_is_zero_when_types_agree():
    same = {c: (np.array([5, 5, 5, 5.]), np.array([5, 5, 5, 5.])) for c in range(3)}
    assert bb.masking_statistic(same) == 0.0
    diff = {0: (np.array([20, 0, 0, 0.]), np.array([0, 20, 0, 0.]))}
    assert bb.masking_statistic(diff) > 20


def test_thinking_always_on_models_get_low_effort_and_no_sampling_params():
    class Probe:
        def __init__(self): self.seen = {}
        def create(self, *, model, max_tokens, messages, system=None, thinking=None,
                   output_config=None, extra_body=None, **kw):
            self.seen = dict(max_tokens=max_tokens, thinking=thinking,
                             output_config=output_config, extra_body=extra_body)
            return types.SimpleNamespace(content=[text_block("x")], stop_reason="end_turn")
    probe = Probe()
    ep = AnthropicEndpoint(model="claude-fable-5-1",
                           client=types.SimpleNamespace(messages=probe))
    ep._one("hi")
    assert probe.seen["thinking"] is None          # explicit disabled would 400
    assert probe.seen["output_config"] == {"effort": "low"}
    assert probe.seen["extra_body"] is None         # no temperature sent at all
    assert probe.seen["max_tokens"] >= 1024         # room for thinking + answer


if __name__ == "__main__":
    import inspect
    import sys
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and inspect.isfunction(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as exc:  # noqa: BLE001 -- report and continue
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
