"""thought-steering pairs | fit | chat"""

import argparse
import sys
import time
from pathlib import Path

from .chat import BLOCK, MAX_NEW_TOKENS, SAMPLER
from .graph import VARIANTS
from .pairs import ATTRIBUTES

NAMES = ["reframe", *ATTRIBUTES]


def strength_term(text):
    name, _, value = text.partition("=")
    if name not in ATTRIBUTES or not value:
        raise argparse.ArgumentTypeError(f"expected NAME=STRENGTH with NAME one of {', '.join(ATTRIBUTES)}")
    return name, float(value)


def chat(args):
    from .chat import reply, steering
    from .graph import Runner
    from .prompt import Tokenizer

    terms = [(Path(args.fits) / name, s) for name, s in [("reframe", args.strength), *args.attribute]]
    rows = steering(terms, args.blocks)
    tokenizer, runner, messages = Tokenizer(), Runner(args.variant), []
    options = {"seed": args.seed, "max_new_tokens": args.max_tokens}
    message = args.message or ask()
    while message:  # each reply sees the conversation so far
        messages.append({"role": "user", "content": message})
        text = ""
        for piece in reply(runner, tokenizer, messages, rows, **options):
            print(piece, end="", flush=True)
            text += piece
        print()
        if args.message:
            break
        messages.append({"role": "assistant", "content": text})
        message = ask()
    return 0


def ask():
    """The next message, or "" at an empty line or the end of input."""
    try:
        return input("> ").strip()
    except EOFError:
        return ""


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # replies can hold emoji, and the Windows console default is cp1252
    parser = argparse.ArgumentParser(prog="thought-steering", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pairs = commands.add_parser("pairs", help="download Cognitive Reframing Pairs and write the contrast data")
    pairs.add_argument("--out", default="data")
    fitting = commands.add_parser("fit", help="fit directions on the steered graph (minutes each on a CPU)")
    fitting.add_argument("names", nargs="*", choices=NAMES, metavar="NAME",
                         help=f"directions to fit (default: reframe), from {', '.join(NAMES)}")
    fitting.add_argument("--data", default="data", help="the folder `pairs` wrote")
    fitting.add_argument("--out", default="fits", help="writes OUT/NAME/fit.safetensors and fit.json")
    talking = commands.add_parser("chat", help="steered replies; without a message, a conversation")
    talking.add_argument("message", nargs="?")
    talking.add_argument("--strength", type=float, default=0.3,
                         help="the reframe direction's length as a fraction of the block's median activation norm; "
                              "negative points away from the reframes (default 0.3)")
    talking.add_argument("--blocks", type=int, nargs="+", default=[BLOCK],
                         help=f"one-based blocks to steer (default {BLOCK})")
    talking.add_argument("--attribute", type=strength_term, action="append", default=[], metavar="NAME=STRENGTH",
                         help="also steer along an attribute's direction, for example empathy=0.2")
    talking.add_argument("--fits", default="fits")
    talking.add_argument("--seed", type=int, default=SAMPLER["seed"])
    talking.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    for sub in (fitting, talking):
        sub.add_argument("--variant", default="q4f16", choices=VARIANTS,
                         help="the 4-bit graph: q4f16 (default) or q4, which computes in 32-bit floats")
    args = parser.parse_args()
    if args.command == "pairs":
        from .pairs import build
        build(args.out)
    elif args.command == "fit":
        from .fit import fit
        from .graph import Runner
        from .prompt import Tokenizer
        tokenizer, runner = Tokenizer(), Runner(args.variant)  # built once for every name
        for name in args.names or ["reframe"]:
            start, out = time.perf_counter(), Path(args.out) / name
            manifest = fit(Path(args.data) / f"{name}.jsonl", out, args.variant, tokenizer, runner)
            print(f"{out / 'fit.safetensors'}: {manifest['rows']} rows, {manifest['forwards']} texts read, "
                  f"{time.perf_counter() - start:.0f} s")
    else:
        return chat(args)
    return 0
