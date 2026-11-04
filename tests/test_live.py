"""Checks against the live page's files. The steered-model test downloads the 0.58 GB model once."""

import json
from pathlib import Path

import numpy as np
import pytest

from thought_steering import graph
from thought_steering.chat import sampler
from thought_steering.fit import data_sha256
from thought_steering.pairs import build
from thought_steering.prompt import Tokenizer

HERE = Path(__file__).parent
# SHA-256 of the graphs the page serves (decoder_model_merged_*_steered_fafab72d_s5.onnx).
LIVE_GRAPHS = {"q4f16": "c4cd3acd70d4175c91caecb7a0f6e6627a6f2b9b577fe149bf8d1dd4496145f3",
               "q4": "2911e7764302ee7827a9667340a7ad7ac09c1190412e9b91a08a75d9ff4ecb0a"}
# The data_sha256 of each live fit's fit.json.
LIVE_DATA = {
    "reframe": "6f79dcb5de6e394622969ba9f3900c683472f8f425810322723c6ad437db2418",
    "actionability": "6e066e05c1d5d763cd45670c991ea909e19b611ae71a003203049ef904004072",
    "empathy": "9aa656da6a45cd163f958a3307650c8d7f1fe3734746b8884b340543211a94a8",
    "positivity": "86d1234c3f650de0d1af2c07164cc46bf73a7ce01549a84b4abf013dfa7d7bf8",
    "rationality": "4a8719a6f21935cfc8dba7b39555645dfda9617a530d8cc3888883fbfb818d15",
    "readability": "2dbba25084a67e7c55c4bc6c0ecccef08c0b7444708a48fdd623388ef6fee90b",
    "specificity": "996644b267af1fda17aa0582fb626cd47a803b8395945d04f96057db2ddc57ee",
    "thinking_trap_addressing": "75258ee0abe992f81cee173603c9f0baa43c46be1debaf879ebd7ec565f74498",
}


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()


@pytest.mark.parametrize("variant", graph.VARIANTS)
def test_steered_graph_is_the_live_one(variant):
    assert graph.live_sha256(graph.steered_decoder(variant), variant) == LIVE_GRAPHS[variant]


def test_contrast_data_is_what_the_live_fits_read(tmp_path):
    """The files `pairs` writes hash, as fit.json records them, to the live fits' data_sha256."""
    for path in build(tmp_path):
        assert data_sha256(path.read_bytes()) == LIVE_DATA[path.stem], path.stem


def test_sampler_draws_the_page_tokens():
    """Tokens the page's sampler (model.mjs) drew from the same logits."""
    page = {(0, 1.0, 20, 1.0, 2.0): [1419, 304, 2584, 790, 1130, 597, 2586, 2344, 259, 2248, 1133, 1279],
            (7, 0.7, 50, 0.9, 1.5): [837, 2826, 511, 1469, 1324, 1470, 2198, 2053, 2975, 369, 1715, 600],
            (123, 1.3, 5, 0.5, -1.0): [1128, 304, 2293, 499, 2488, 985, 2683, 1180, 2878, 2345, 1812, 2540]}
    i = np.arange(3000)
    for (seed, temperature, top_k, top_p, penalty), expected in page.items():
        pick = sampler(temperature, top_k, top_p, penalty, seed)
        logits = lambda s: (((i * 7919 + s * 104729) % 97) / 8 + ((i * 31) % 13) / 1024).astype(np.float32)  # noqa: E731
        assert [pick(logits(s)) for s in range(len(expected))] == expected


def test_steered_model_matches_the_page_self_check(tokenizer):
    """The page's load-time reference (check-q4f16.json): for "Hi", the prompt's ids, the first 64 dimensions of each
    block output at its last token, and the first 8 greedy tokens."""
    check = json.loads((HERE / "check-q4f16.json").read_text(encoding="utf-8"))
    ids = tokenizer.prompt([{"role": "user", "content": check["prompt"]}])
    assert ids == check["ids"]
    runner = graph.Runner("q4f16")
    logits, blocks, state = runner.prefill(ids)
    np.testing.assert_allclose(blocks[:, :64], check["blocks"], atol=1e-4)
    tokens = []
    for _ in check["tokens"]:
        tokens.append(int(logits.argmax()))
        logits, _, state = runner.step(tokens[-1:], state)
    assert tokens == check["tokens"]
