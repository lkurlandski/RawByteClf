"""
Useful printing of various models.
"""

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


def get_hrrformer_configs(max_length: int = 2 ** 17):
    configs = [dict(max_position_embeddings=max_length, vocab_size=264, embedding_size=256, hidden_size=256, intermediate_size=512, num_attention_heads=8, num_hidden_layers=1)]
    for v in [264]:
        for h in [128, 256, 384, 512, 768, 1024]:
            for l in [1, 2, 4, 6, 8, 12]:
                for n in [1, 2, 4, 8, 12, 16]:
                    config = dict(
                        max_position_embeddings=max_length,
                        vocab_size=v,
                        embedding_size=h,
                        hidden_size=h,
                        intermediate_size=4 * h,
                        num_attention_heads=n,
                        num_hidden_layers=l,
                    )
                    configs.append(config)
    return configs


def get_malconv_configs():
    configs = [
        dict(vocab_size=264, embedding_size=8, channels=128, stride=500, kernel_size=500),
        dict(vocab_size=264, embedding_size=256, channels=128, stride=500, kernel_size=500),
    ]
    for v in [264]:
        for e in [128, 256, 384, 512, 768, 1024]:
            for c in [64, 128, 256]:
                for s in [256, 512, 768]:
                    for k in [256, 512, 768]:
                        config = dict(
                            vocab_size=v,
                            embedding_size=e,
                            channels=c,
                            stride=s,
                            kernel_size=k,
                        )
                        configs.append(config)
    return configs


def get_malconv2_configs():
    configs = [
        dict(vocab_size=264, embedding_size=8, channels=256, stride=64, kernel_size=256),
        dict(vocab_size=264, embedding_size=256, channels=256, stride=64, kernel_size=256),
    ]
    for v in [264]:
        for e in [128, 256, 384, 512, 768, 1024]:
            for c in [64, 128, 256]:
                for s in [64, 128, 256]:
                    for k in [256, 512, 768]:
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
    for v in [264]:
        for h in [128, 256, 384, 512, 768, 1024]:
            for l in [2, 4, 6, 8, 10, 12, 16]:
                config = dict(
                    vocab_size=v,
                    embedding_size=8,
                    hidden_size=h,
                    num_hidden_layers=l,
                )
                configs.append(config)
    return configs


def display():
    with open(OUTFILE, "r") as fp:
        for line in fp:
            d = json.loads(line.strip())
            for k in d:
                if k[0:2] == "p_":
                    d[k] = str(round(d[k] / 1e6, 2)) + "M"
            if "hidden_size" in d and "num_hidden_layers" in d:
                d["shape"] = f"{round(d['hidden_size'] / d['num_hidden_layers'])}"
            # if "hidden_size" in d:
            #     d.pop("embedding_size")
            if "vocab_size" in d:
                d.pop("vocab_size")
            print(d)


def run():
    NUM_LABELS = 256
    N_HRRFORMER = 0
    N_MALCONV = 0
    N_MALCONV2 = 0
    N_MAMBA = None

    for d in tqdm(get_hrrformer_configs()[0:N_HRRFORMER], desc="HRRFormer..."):
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

    for d in tqdm(get_malconv2_configs()[0:N_MALCONV2], desc="MalConv2..."):
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
    # run()
    display()
