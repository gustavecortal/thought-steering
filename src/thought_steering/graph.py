"""The page's steered decoder: onnx-community's 4-bit Qwen3.5-0.8B graph with a `steering` input and a
`block_outputs` output, built by graph surgery, and a runner for it in ONNX Runtime on the CPU.

The graph fuses each residual addition into the next RMSNorm (a SkipSimplifiedLayerNormalization node: output 0 is
normalized, output 3 is the sum of its input and skip input). Block l's output, the residual stream after block l,
is that sum inside block l+1's input norm, or inside the final norm for the last block. The surgery adds
`steering[l]` to that node's skip input, so the sum and its normalized copy both carry it, and returns the sums at
the last position of each call as `block_outputs` [batch, blocks, hidden].
"""

import os
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper, numpy_helper

REPO = "onnx-community/Qwen3.5-0.8B-ONNX-OPT"
REVISION = "fafab72d87a9e6be3925b38caf48286d2838f2d0"
VARIANTS = ["q4f16", "q4"]  # q4f16 computes in 16-bit floats, and the page uses q4 on GPUs without them
BLOCKS, WIDTH = 24, 1024
SURGERY = 5  # the version of steered_decoder, recorded in the fits and in the live graph's file name
# The model's files as plain files in one folder: ONNX Runtime refuses external weights reached through the symbolic
# links of the Hugging Face cache, which point outside the model's folder on Linux and macOS.
MODELS = Path(os.environ.get("THOUGHT_STEERING_MODELS", Path.home() / ".cache" / "thought-steering")) / REVISION


def fetch(name):
    """A file of the ONNX repository and its external weights (name + "_data"); returns the file's path."""
    hf_hub_download(REPO, name + "_data", revision=REVISION, local_dir=MODELS)
    return hf_hub_download(REPO, name, revision=REVISION, local_dir=MODELS)


def steered_decoder(variant="q4f16", path=None):
    """The steered decoder graph, without loading its weights: they stay in the upstream external-data file, at the
    same offsets."""
    path = path or hf_hub_download(REPO, f"onnx/decoder_model_merged_{variant}.onnx", revision=REVISION, local_dir=MODELS)
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    graph.input.append(helper.make_tensor_value_info("steering", TensorProto.FLOAT, [BLOCKS, WIDTH]))
    graph.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), "/steer/last"))
    # One Gather per row, not a 24-way Split: WebGPU allows 16 storage buffers per shader (one per tensor).
    graph.initializer.extend(numpy_helper.from_array(np.array(l, np.int64), f"/steer/{l}/index") for l in range(BLOCKS))
    sites = {f"/model/layers.{l + 1}/input_layernorm/SkipLayerNorm": l for l in range(BLOCKS - 1)}
    sites[f"/model/layers.{BLOCKS}/final_norm_layernorm/SkipLayerNorm"] = BLOCKS - 1
    rows = [f"/steer/{l}/row" for l in range(BLOCKS)]
    nodes = [helper.make_node("Gather", ["steering", f"/steer/{l}/index"], [rows[l]], axis=0) for l in range(BLOCKS)]
    sums = {}
    for node in graph.node:
        if node.name in sites:
            l = sites[node.name]
            nodes += [helper.make_node("CastLike", [rows[l], node.input[1]], [f"/steer/{l}/cast"]),
                      helper.make_node("Add", [node.input[1], f"/steer/{l}/cast"], [f"/steer/{l}/skip"])]
            node.input[1] = f"/steer/{l}/skip"
            node.output.extend([""] * (4 - len(node.output)))  # the final norm does not expose its sum by default
            if not node.output[3]:
                node.output[3] = f"/steer/{l}/sum"
            sums[l] = node.output[3]
        nodes.append(node)
    if len(sums) != BLOCKS:
        raise ValueError(f"found {len(sums)} of {BLOCKS} block sites; the graph layout changed")
    # Each block's sum at the last position, [batch, 1, hidden], concatenated in groups of 8 (the same WebGPU limit).
    kept = [f"/steer/{l}/last" for l in range(BLOCKS)]
    nodes += [helper.make_node("Gather", [sums[l], "/steer/last"], [kept[l]], axis=1) for l in range(BLOCKS)]
    groups = [kept[start:start + 8] for start in range(0, BLOCKS, 8)]
    nodes += [helper.make_node("Concat", group, [f"/steer/group{g}"], axis=1) for g, group in enumerate(groups)]
    nodes += [helper.make_node("Concat", [f"/steer/group{g}" for g in range(len(groups))], ["/steer/blocks"], axis=1),
              helper.make_node("Cast", ["/steer/blocks"], ["block_outputs"], to=TensorProto.FLOAT)]
    graph.output.append(helper.make_tensor_value_info("block_outputs", TensorProto.FLOAT, ["batch_size", BLOCKS, WIDTH]))
    del graph.node[:]
    graph.node.extend(nodes)
    return model


class Runner:
    """The steered decoder and its token embedding in ONNX Runtime (CPU), with explicit recurrent and attention
    state, as the page runs them in the browser."""

    def __init__(self, variant="q4f16", threads=0):
        import onnxruntime as ort

        self.variant = variant

        def session(model, folder):
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            # A graph given as bytes finds its external weights in this folder.
            options.add_session_config_entry("session.model_external_initializers_file_folder_path", folder)
            return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])

        decoder = fetch(f"onnx/decoder_model_merged_{variant}.onnx")
        embed = fetch(f"onnx/embed_tokens_{variant}.onnx")
        folder = str(Path(decoder).parent)
        self.decoder = session(steered_decoder(variant, decoder).SerializeToString(), folder)
        self.embed = session(embed, folder)
        self.inputs = {i.name: i for i in self.decoder.get_inputs()}
        self.outputs = [o.name for o in self.decoder.get_outputs()]
        self.state_names = {n: n.replace("present_", "past_").replace("present.", "past_key_values.")
                            for n in self.outputs if n.startswith("present")}

    def empty(self):
        """The state before any token: no past tokens, batch 1."""
        state = {}
        for name, info in self.inputs.items():
            if name.startswith("past_"):
                shape = [0 if d == "past_sequence_length" else 1 if isinstance(d, str) else d for d in info.shape]
                state[name] = np.zeros(shape, np.float16 if "float16" in info.type else np.float32)
        return state

    def step(self, ids, state, steering=None):
        """Decodes ids after state, with steering [blocks, hidden] added at every position (zero when None).
        Returns the last position's logits [vocab] and block outputs [blocks, hidden], and the next state."""
        past = next((v.shape[2] for k, v in state.items() if k.startswith("past_key_values")), 0)
        embeds = self.embed.run(None, {"input_ids": np.array([ids], np.int64)})[0]
        positions = np.arange(past, past + len(ids), dtype=np.int64)
        feed = {"inputs_embeds": embeds, "attention_mask": np.ones((1, past + len(ids)), np.int64),
                "position_ids": np.broadcast_to(positions, (3, 1, len(ids))).copy(), **state,
                "num_logits_to_keep": np.array(1, np.int64),
                "steering": np.zeros((BLOCKS, WIDTH), np.float32) if steering is None else steering}
        values = dict(zip(self.outputs, self.decoder.run(None, feed)))
        following = {self.state_names[n]: values[n] for n in self.state_names}
        return values["logits"][0, -1].astype(np.float32), values["block_outputs"][0], following

    def prefill(self, ids, steering=None):
        """The prompt: unsteered except its last token, as on the page."""
        state = self.empty()
        if len(ids) > 1:
            _, _, state = self.step(ids[:-1], state)
        return self.step(ids[-1:], state, steering)
