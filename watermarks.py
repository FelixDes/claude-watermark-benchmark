"""SynthID variants for the local reference model.

`transformers` ships SynthID-Text with repeated-context masking (K=1) whose
history starts *empty*: `context` is zeros and `context_history` is zeros, and
both fill up one generated token at a time. The prompt never enters either --
not even its last token. So a context that already occurs in the prompt is
"new" to the watermark, and the model's echo of it gets watermarked.

That is the property the paper's black-box test leans on, and the one a
deployer has every reason to remove: a model that reproduces the user's text
should not stamp it as its own. `PromptAwareSynthIDProcessor` is the smallest
such change -- seed the history with every h-gram of the prompt before the
first step -- so the effect on detection tests can be measured in isolation.
"""

from __future__ import annotations

import torch
from transformers import SynthIDTextWatermarkLogitsProcessor


class PromptAwareSynthIDProcessor(SynthIDTextWatermarkLogitsProcessor):
    """SynthID whose repeated-context history includes the prompt.

    Everything else -- keys, g-values, tournament, K=1 -- is inherited
    unchanged. The only difference is what counts as "seen before".
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.state is None:
            # First step: `input_ids` is exactly the prompt.
            self._init_state(input_ids.shape[0])
            self._seed_from_prompt(input_ids)
        return super().__call__(input_ids, scores)

    def _seed_from_prompt(self, prompt: torch.LongTensor) -> None:
        h = self.ngram_len - 1
        batch, length = prompt.shape
        ones = torch.ones(batch, device=self.device, dtype=torch.long)

        # Every window of h tokens in the prompt, hashed exactly as the base
        # class hashes a generation-time context, oldest first.
        hashes = [
            self.accumulate_hash(ones, prompt[:, i - h:i])
            for i in range(h, length + 1)
        ]
        if hashes:
            hist = torch.stack(hashes, dim=1)
            room = self.state.context_history.shape[1]
            n = min(hist.shape[1], room)
            self.state.context_history[:, :n] = hist[:, -n:]

        # The base class appends `input_ids[:, -1]` on this same call and drops
        # the oldest slot, so seeding with prompt[-h-1:-1] leaves the context
        # at prompt[-h:] -- the real preceding tokens, not zeros.
        if length >= h + 1:
            self.state.context = prompt[:, -h - 1:-1].clone()
