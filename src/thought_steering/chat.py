"""Steered replies with the page's sampler."""

import math

import numpy as np

# The page's defaults.
SAMPLER = {"temperature": 1.0, "top_k": 20, "top_p": 1.0, "presence_penalty": 2.0, "seed": 0}
MAX_NEW_TOKENS = 256


def mulberry32(seed):
    """The page's seeded random numbers in [0, 1)."""
    state = seed & 0xFFFFFFFF
    mask = 0xFFFFFFFF

    def random():
        nonlocal state
        state = (state + 0x6D2B79F5) & mask
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & mask
        t ^= (t + ((t ^ (t >> 7)) * (t | 61))) & mask
        return (t ^ (t >> 14)) / 4294967296

    return random


def sampler(temperature, top_k, top_p, presence_penalty, seed):
    """pick(logits) → token: a presence penalty on every token already picked, then temperature, top-k, top-p and
    a seeded draw (llama.cpp's order), as on the page. logits is changed in place."""
    seen, random = set(), mulberry32(seed)

    def pick(logits):
        for token in seen:
            logits[token] -= presence_penalty
        # The top_k largest in the order of a stable sort of the whole vocabulary (ties keep the lower token id
        # first), sorting only the tokens at or above the k-th largest value.
        k = min(top_k, logits.size)
        kept = np.flatnonzero(logits >= np.partition(logits, logits.size - k)[logits.size - k])
        top = kept[np.argsort(-logits[kept], kind="stable")][:top_k]
        values = logits[top].astype(np.float64)
        weights = [math.exp((v - values[0]) / temperature) for v in values]
        total = sum(weights)
        kept, mass = 0, 0.0
        while kept < len(top) and mass < top_p:
            mass += weights[kept] / total
            kept += 1
        r, i = random() * mass, 0
        while i < kept - 1:
            r -= weights[i] / total
            if r <= 0:
                break
            i += 1
        token = int(top[i])
        seen.add(token)
        return token

    return pick


def reply(runner, tokenizer, messages, rows, max_new_tokens=MAX_NEW_TOKENS, **options):
    """Yields the reply's text piece by piece: the prompt runs unsteered except its last token, and every token the
    model writes is steered by rows."""
    pick, text = sampler(**{**SAMPLER, **options}), tokenizer.stream()
    logits, _, state = runner.prefill(tokenizer.prompt(messages), rows)
    for n in range(1, max_new_tokens + 1):
        token = pick(logits)
        if token == tokenizer.eos:
            return
        piece = text(token)
        if piece:
            yield piece
        if n < max_new_tokens:
            logits, _, state = runner.step([token], state, rows)
