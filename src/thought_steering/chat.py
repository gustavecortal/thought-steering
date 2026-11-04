"""Steered replies with the page's sampler."""

import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from .graph import BLOCKS, WIDTH

# The page's defaults.
SAMPLER = {"temperature": 1.0, "top_k": 20, "top_p": 1.0, "presence_penalty": 2.0, "seed": 0}
MAX_NEW_TOKENS = 256
BLOCK = 12  # one-based


def steering(terms, blocks=(BLOCK,)):
    """The additions [blocks, hidden] of fits at strengths, terms = [(fit folder, strength)]: strength * R[l] *
    v[l] / |v[l]| at each chosen one-based block, summed over fits; None when every strength is 0. Each fit's term is
    added in double precision and the sum rounded to float32, as the page's Float32Array does."""
    if not blocks or len(set(blocks)) != len(blocks) or not all(1 <= b <= BLOCKS for b in blocks):
        raise ValueError(f"choose distinct blocks from 1 to {BLOCKS}")
    if all(strength == 0 for _, strength in terms):
        return None
    rows = np.zeros((BLOCKS, WIDTH), np.float32)
    for folder, strength in terms:
        fit = load_file(Path(folder) / "fit.safetensors")
        scale = fit["reference"].astype(np.float64) / fit["norms"]
        vectors = (fit["directions"] * scale[:, None]).astype(np.float32)
        for b in blocks:
            rows[b - 1] = rows[b - 1] + strength * vectors[b - 1].astype(np.float64)
    return rows


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
        top = np.argsort(-logits, kind="stable")[:top_k]  # ties keep the lower token id first
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
