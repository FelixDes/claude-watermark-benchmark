"""Minimal SynthID-Text demo: small causal LM on CUDA, watermarked vs plain generation.

Watermarking is applied at sampling time via SynthIDTextWatermarkingConfig, and
scored back with the training-free "mean g-value" detector. Under the null
(no watermark) the mean g-value is ~0.5; watermarked text scores above it.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"

# Watermark parameters. `keys` is the secret: detection only works with the
# exact same list, and its length is the tournament depth.
WATERMARK_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]
NGRAM_LEN = 5

PROMPTS = [
    "Explain in one paragraph why the sky looks blue.",
    "Write a short paragraph about the history of the printing press.",
    "Describe how a heat pump works, in one paragraph.",
]


def build_inputs(tokenizer, prompts):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    return tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)


@torch.no_grad()
def generate(model, inputs, watermarking_config, max_new_tokens=180):
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        max_new_tokens=max_new_tokens,
        watermarking_config=watermarking_config,
    )
    # Keep only the newly generated tokens — the watermark lives there.
    return out[:, inputs["input_ids"].shape[1]:]


@torch.no_grad()
def mean_g_value(processor, tokens, eos_token_id):
    """Training-free detector score, per sequence. ~0.5 = no watermark."""
    g_values = processor.compute_g_values(input_ids=tokens).float()  # [B, T-(n-1), depth]
    # Drop n-grams whose context was already seen (their g-values are biased)
    # and everything at/after EOS padding.
    repetition_mask = processor.compute_context_repetition_mask(input_ids=tokens)
    eos_mask = processor.compute_eos_token_mask(
        input_ids=tokens, eos_token_id=eos_token_id
    )[:, NGRAM_LEN - 1:]
    mask = (repetition_mask * eos_mask).float().unsqueeze(-1)

    n_scored = mask.squeeze(-1).sum(dim=1)
    valid = n_scored * g_values.shape[-1]
    total = (g_values * mask).sum(dim=(1, 2))
    scores = torch.where(
        valid > 0, total / valid.clamp(min=1), torch.full_like(total, float("nan"))
    )
    return scores, n_scored


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE
    ).eval()

    watermarking_config = SynthIDTextWatermarkingConfig(
        keys=WATERMARK_KEYS, ngram_len=NGRAM_LEN
    )
    # Same construction the generate() call does internally, so the detector
    # reproduces the exact g-value function used at sampling time.
    detector = SynthIDTextWatermarkLogitsProcessor(
        **watermarking_config.to_dict(), device=DEVICE
    )

    inputs = build_inputs(tokenizer, PROMPTS)
    eos_id = tokenizer.eos_token_id

    watermarked = generate(model, inputs, watermarking_config)
    plain = generate(model, inputs, None)

    wm_scores, wm_n = mean_g_value(detector, watermarked, eos_id)
    pl_scores, pl_n = mean_g_value(detector, plain, eos_id)

    for i, prompt in enumerate(PROMPTS):
        print("=" * 100)
        print(f"PROMPT: {prompt}")
        print(f"\n-- watermarked (score={wm_scores[i]:.4f}, scored n-grams={int(wm_n[i])}) --")
        print(tokenizer.decode(watermarked[i], skip_special_tokens=True).strip())
        print(f"\n-- plain (score={pl_scores[i]:.4f}, scored n-grams={int(pl_n[i])}) --")
        print(tokenizer.decode(plain[i], skip_special_tokens=True).strip())

    print("=" * 100)
    print(f"mean g-value  watermarked: {wm_scores.nanmean():.4f}")
    print(f"mean g-value  plain      : {pl_scores.nanmean():.4f}   (null expectation ~0.5)")


if __name__ == "__main__":
    main()
