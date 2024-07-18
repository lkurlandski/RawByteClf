"""
Useful printing of various models.
"""

from argparse import ArgumentParser
import json
from pprint import pformat, pprint
import os
import sys
from typing import Optional

from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils import count_parameters
from src.architectures.hrrformer import HRRConfig, HRRForSequenceClassification
from src.architectures.malconv_hf import MalConvConfig, MalConvForSequenceClassification
from src.architectures.malconv2 import MalConv2Config, MalConv2ForSequenceClassification
from src.architectures.mamba_hf import MambaConfig, MambaForSequenceClassification


OUTFILE = "./tmp/model_sizes.jsonl"
MAX_LENGTH = 16384
NUM_LABELS = 256
N_HRRFORMER = None
N_MALCONV = None
N_MALCONV2 = None
N_MAMBA = None


VOCAB_SIZES = [256 + 8, 4096 + 8, 16384 + 8, 65536 + 8]
HIDDEN_SIZES = [128, 256, 384, 512, 768, 1024]
NUM_HIDDEN_LAYERS = [1, 2, 4, 6, 8, 12, 16]
EMBEDDING_SIZES = [8, 16, 32, 64] + HIDDEN_SIZES
MODES = ["uni", "bi"]
NUM_ATTENTION_HEADS = [1, 2, 4, 8, 16]
CHANNELS = [64, 128, 256]
STRIDES = [64, 128, 256, 512, 1024]
KERNEL_SIZES = [64, 128, 256, 512, 1024]


def get_hrrformer_configs(max_length: int):
    configs = []
    for v in VOCAB_SIZES:
        for h in HIDDEN_SIZES:
            for l in NUM_HIDDEN_LAYERS:
                for n in NUM_ATTENTION_HEADS:
                    for e in EMBEDDING_SIZES:
                        config = dict(
                            max_position_embeddings=max_length,
                            vocab_size=v,
                            hidden_size=h,
                            intermediate_size=4 * h,
                            num_attention_heads=n,
                            num_hidden_layers=l,
                            embedding_size=e,
                        )
                        configs.append(config)
    return configs


def get_malconv_configs():
    configs = []
    for v in VOCAB_SIZES:
        for c in CHANNELS:
            for s in STRIDES:
                for k in KERNEL_SIZES:
                    for e in EMBEDDING_SIZES:
                        config = dict(
                            vocab_size=v,
                            embedding_size=e,
                            channels=c,
                            stride=s,
                            kernel_size=k,
                        )
                        configs.append(config)
    return configs


def get_mamba_configs():
    configs = []
    for m in MODES:
        for v in VOCAB_SIZES:
            for h in HIDDEN_SIZES:
                for l in NUM_HIDDEN_LAYERS:
                    for e in EMBEDDING_SIZES:
                        config = dict(
                            mode=m,
                            vocab_size=v,
                            hidden_size=h,
                            num_hidden_layers=l,
                            embedding_size=e,
                        )
                        configs.append(config)

    return configs


def display():
    with open(OUTFILE, "r") as fp:
        for line in fp:
            d = json.loads(line.strip())

            d["arch"] = d.pop("architecture")
            if "mode" in d:
                d["mode"] = d.pop("mode")

            if "vocab_size" in d:
                d["V"] = d.pop("vocab_size")
            if "hidden_size" in d:
                d["H"] = d.pop("hidden_size")
            if "num_hidden_layers" in d:
                d["L"] = d.pop("num_hidden_layers")

            if "intermediate_size" in d:
                d["I"] = d.pop("intermediate_size")
            if "num_attention_heads" in d:
                d["N"] = d.pop("num_attention_heads")
            if "max_position_embeddings" in d:
                d["M"] = d.pop("max_position_embeddings")

            if "channels" in d:
                d["C"] = d.pop("channels")
            if "kernel_size" in d:
                d["K"] = d.pop("kernel_size")
            if "stride" in d:
                d["S"] = d.pop("stride")

            if "embedding_size" in d:
                d["E"] = d.pop("embedding_size")

            for k in list(d.keys()):
                if k[0:2] == "p_":
                    v = d[k]
                    d.pop(k)
                    d[k] = str(round(v / 1e6, 2)) + "M"

            print(d)


def run():

    for d in tqdm(get_hrrformer_configs(MAX_LENGTH)[0:N_HRRFORMER], desc="HRRFormer..."):
        config = HRRConfig(num_labels=NUM_LABELS, **d)
        model = HRRForSequenceClassification(config)
        p_total = count_parameters(model)
        p_embd = count_parameters(model.bert.embeddings)
        p_backbone = count_parameters(model.bert) - p_embd
        p_clf = count_parameters(model.classifier)
        counts = {"p_total": p_total, "p_embd": p_embd, "p_backbone": p_backbone, "p_clf": p_clf}
        out = {"architecture": "hrrformer"} | d | counts
        with open(OUTFILE, "a") as fp:
            fp.write(json.dumps(out) + "\n")

    for d in tqdm(get_malconv_configs()[0:N_MALCONV], desc="MalConv..."):
        config = MalConvConfig(num_labels=NUM_LABELS, **d)
        model = MalConvForSequenceClassification(config)
        p_total = count_parameters(model)
        p_embd = count_parameters(model.malconv.embed)
        p_backbone = count_parameters(model.malconv) - p_embd
        p_clf = count_parameters(model.clf_head)
        counts = {"p_total": p_total, "p_embd": p_embd, "p_backbone": p_backbone, "p_clf": p_clf}
        out = {"architecture": "malconv"} | d | counts
        with open(OUTFILE, "a") as fp:
            fp.write(json.dumps(out) + "\n")

    for d in tqdm(get_malconv_configs()[0:N_MALCONV2], desc="MalConv2..."):
        config = MalConv2Config(num_labels=NUM_LABELS, **d)
        model = MalConv2ForSequenceClassification(config)
        p_total = count_parameters(model)
        p_embd = count_parameters(model.malconv.malconv.embd)
        p_backbone = count_parameters(model.malconv.malconv) - p_embd
        p_clf = count_parameters(model.malconv.malconv.fc_1) + count_parameters(model.malconv.malconv.fc_2)
        counts = {"p_total": p_total, "p_embd": p_embd, "p_backbone": p_backbone, "p_clf": p_clf}
        out = {"architecture": "malconv2"} | d | counts
        with open(OUTFILE, "a") as fp:
            fp.write(json.dumps(out) + "\n")

    for d in tqdm(get_mamba_configs()[0:N_MAMBA], desc="Mamba..."):
        config = MambaConfig(num_labels=NUM_LABELS, **d)
        model = MambaForSequenceClassification(config)
        p_total = count_parameters(model)
        p_embd = count_parameters(model.backbone.embeddings)
        p_backbone = count_parameters(model.backbone) - p_embd
        p_clf = count_parameters(model.clf_head)
        counts = {"p_total": p_total, "p_embd": p_embd, "p_backbone": p_backbone, "p_clf": p_clf}
        out = {"architecture": "mamba"} | d | counts
        with open(OUTFILE, "a") as fp:
            fp.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    if args.run:
        run()
    if args.display:
        display()
