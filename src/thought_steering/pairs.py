"""Contrast data from Cognitive Reframing Pairs: one file for the reframe direction and one per reframing attribute.

A row holds a chat context, a negative side and a positive side. A side is a text or a list of sides, and the fit
averages each list, so nesting sets the weights (fit.py). Each text is read as the model's reply to the context.
"""

import json
import random
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from huggingface_hub import hf_hub_download

from .prompt import Tokenizer

DATASET = "gustavecortal/cognitive-reframing-pairs"
DATASET_REVISION = "783519aad5dea37843a95d45c98afdd82bbb4b5b"
# The request each text answers: a thought about the situation.
THOUGHT = ("Situation: {situation}\n\nWrite one plausible first-person thought someone might have in this situation. "
           "Use one or two sentences of self-talk. Return only the thought.")
# The reframing attributes of Sharma et al. (2023), as the dataset varies them.
ATTRIBUTES = ["actionability", "empathy", "positivity", "rationality", "readability", "specificity",
              "thinking_trap_addressing"]


def read_pairs():
    """The dataset's pairs, each with its situation and thought as {id, text} (and the situation's domain)."""
    path = hf_hub_download(DATASET, "pairs.jsonl", repo_type="dataset", revision=DATASET_REVISION)
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            p = json.loads(line)
            pairs.append({"id": p["pair_id"], "attribute": p["attribute"], "less": p["less"], "more": p["more"],
                          "situation": {"id": p["situation_id"], "domain": p["domain"], "text": p["situation"]},
                          "thought": {"id": p["thought_id"], "text": p["thought"]}})
    return pairs


def context(situation):
    return [{"role": "user", "content": THOUGHT.format(situation=situation["text"])}]


def reframe_rows(pairs, tokenizer, situations=128, seed=0, tolerance=0.15):
    """One row per negative thought, grouped by situation: negative = the thought, positive = its reframes as
    [attribute][pair][kept less and more variants]. The fit then averages variants within a pair, pairs within an
    attribute, attributes within a thought, thoughts within a situation, then situations. A variant is kept when its
    token length is within the tolerance of the thought's, so length does not drive the direction."""
    by_id, thoughts, lengths = {}, {}, {}
    kept = defaultdict(lambda: defaultdict(dict))  # (situation id, thought id) → attribute → pair id → texts
    for pair in pairs:
        situation, thought = pair["situation"], pair["thought"]
        by_id[situation["id"]], thoughts[thought["id"]] = situation, thought["text"]
        length = lengths.setdefault(thought["id"], len(tokenizer.encode(thought["text"])))
        texts = [pair[side] for side in ("less", "more")
                 if abs(len(tokenizer.encode(pair[side])) - length) <= tolerance * length]
        if texts:  # a single kept variant stands in for its pair
            kept[situation["id"], thought["id"]][pair["attribute"]][pair["id"]] = texts

    # Seeded and spread over domains: each domain's situations shuffled, then one per domain in turn.
    domains = defaultdict(list)
    for situation_id in sorted({s for s, _ in kept}):
        domains[by_id[situation_id]["domain"]].append(situation_id)
    rng = random.Random(seed)
    for domain in sorted(domains):
        rng.shuffle(domains[domain])
    chosen = [s for turn in zip_longest(*(domains[d] for d in sorted(domains))) for s in turn if s][:situations]

    rows = []
    for situation_id in chosen:
        for (s, thought_id), attributes in sorted(kept.items()):
            if s != situation_id:
                continue
            ordered = [sorted(by_pair.items()) for _, by_pair in sorted(attributes.items())]
            rows.append({"id": thought_id, "group_id": situation_id,
                         "context": context(by_id[s]), "negative": thoughts[thought_id],
                         "positive": [[texts for _, texts in pairs_] for pairs_ in ordered],
                         "provenance": {"domain": by_id[s]["domain"], "attributes": sorted(attributes),
                                        "pairs": [[pair_id for pair_id, _ in pairs_] for pairs_ in ordered]}})
    return rows


def attribute_rows(pairs, tokenizer, attribute, tolerance=0.25):
    """One row per pair of the attribute, grouped by situation: negative = less, positive = more. A pair is kept
    when its two variants' token lengths are within the tolerance of each other."""
    rows = []
    for pair in pairs:
        if pair["attribute"] != attribute:
            continue
        less, more = (len(tokenizer.encode(pair[side])) for side in ("less", "more"))
        if abs(more / less - 1) <= tolerance:
            rows.append({"id": pair["id"], "group_id": pair["situation"]["id"], "context": context(pair["situation"]),
                         "negative": pair["less"], "positive": pair["more"],
                         "provenance": {"domain": pair["situation"]["domain"], "thought": pair["thought"]["id"]}})
    return rows


def build(out="data"):
    """Writes reframe.jsonl and one <attribute>.jsonl per attribute to out; returns their paths."""
    tokenizer, pairs = Tokenizer(), read_pairs()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sets = {"reframe": reframe_rows(pairs, tokenizer)}
    sets.update((a, attribute_rows(pairs, tokenizer, a)) for a in ATTRIBUTES)
    paths = []
    for name, rows in sets.items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8",
                        newline="\n")
        print(f"{path}: {len(rows)} rows, {len({r['group_id'] for r in rows})} situations")
        paths.append(path)
    return paths
