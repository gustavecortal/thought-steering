"""Qwen3.5-0.8B's tokenizer and chat template, rendered as transformers renders it, so prompts get the same token ids
as on the web page."""

import hashlib
import json
from pathlib import Path

import jinja2
from huggingface_hub import hf_hub_download
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer as Fast
from tokenizers.decoders import DecodeStream

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def environment():
    """The Jinja environment of transformers' apply_chat_template: blocks trimmed, loop controls, a tojson filter
    that does not escape HTML, and raise_exception."""
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    return env


class Tokenizer:
    def __init__(self, model=MODEL, revision=REVISION):
        file = lambda name: Path(hf_hub_download(model, name, revision=revision))  # noqa: E731
        tokenizer = file("tokenizer.json").read_bytes()
        config = json.loads(file("tokenizer_config.json").read_text(encoding="utf-8"))
        template = file("chat_template.jinja").read_text(encoding="utf-8")
        self.fast = Fast.from_str(tokenizer.decode("utf-8"))
        self.template = environment().from_string(template)
        # As in transformers, the special-token strings are template variables.
        self.special = {k: v for k, v in config.items() if k.endswith("_token") and isinstance(v, str)}
        self.eos = self.fast.token_to_id(config["eos_token"])
        self.sha256 = {"template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                       "tokenizer_sha256": hashlib.sha256(tokenizer).hexdigest()}

    def prompt(self, messages):
        """Token ids of the chat ending with the generation prompt, with thinking off (the page's default)."""
        text = self.template.render(messages=messages, tools=None, documents=None, add_generation_prompt=True,
                                    enable_thinking=False, **self.special)
        return self.encode(text)

    def encode(self, text):
        """Token ids without added special tokens (special-token text in the input is still recognized)."""
        return self.fast.encode(text, add_special_tokens=False).ids

    def stream(self):
        """Incremental decoding: step(token) gives the new text, or None inside a character split across tokens."""
        decoder = DecodeStream(skip_special_tokens=True)
        return lambda token: decoder.step(self.fast, token)
