"""

"""

from collections import defaultdict
from copy import deepcopy
from typing import Any

from ray import tune


TunedConfigs = defaultdict(lambda: defaultdict(dict), 
{
    "malconv": defaultdict(dict, {
        # eval_loss = 5.778996
        65536: {
            "channels": 64,
            "chunk_size": 1024,
            "overlap": 256,
            "stride": 448,
            "window_size": 64,
        },
    }),
    "malconvgct": defaultdict(dict, {
        # eval_loss = 5.777025
        65536: {
            "channels": 64,
            "chunk_size": 2048,
            "overlap": 768,
            "stride": 512,
            "window_size": 192,
        },
    }),
    "mymalconv": defaultdict(dict, {
        # eval_loss = 5.396931
        65536: {
            "channels": 192,
            "hidden_size": 128,
            "stride": 512,
            "window_size": 512,
        },
    }),
    "hrrformer": defaultdict(dict, {
        # eval_loss = 
        65536: {
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
        },
        16384: {
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
        },
    }),
    "longformer": defaultdict(dict, {
        # eval_loss = 5.124239 (MLM task)
        16384: {
            "hidden_size": 1024,
            "intermediate_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "attention_window": 128,
        },
    }),
})

TunedConfigs["longformer"][65536] = deepcopy(TunedConfigs["longformer"][16384])


def hp_space_hrrformer(trial: Any) -> dict[str, float | int]:  # pylint: disable=unused-argument

    return {
        "hidden_size": tune.choice([256, 512, 768, 1024]),
        "intermediate_size": tune.choice([512, 1024, 1536, 2048]),
        "num_hidden_layers": tune.choice([1, 2, 3, 4]),
        "num_attention_heads": tune.choice([1, 2, 4, 8]),
    }


def hp_space_longformer(trial: Any) -> dict[str, float | int]:  # pylint: disable=unused-argument
    """
    - hidden_size % num_attention_heads == 0
    """

    return {
        "hidden_size": tune.choice([256, 512, 768, 1024]),
        "intermediate_size": tune.choice([512, 1024, 1536, 2048]),
        "num_hidden_layers": tune.choice([1, 2, 3, 4]),
        "num_attention_heads": tune.choice([1, 2, 4, 8]),
        "attention_window": tune.choice([128, 256, 512, 1024, 2048, 4096]),
    }


def hp_space_malconv(trial: Any) -> dict[str, float | int]:  # pylint: disable=unused-argument
    return {
        "stride": tune.choice([64, 128, 192, 256, 320, 384, 448, 512]),
        "window_size": tune.choice([64, 128, 192, 256, 320, 384, 448, 512]),
        "channels": tune.choice([64, 128, 192]),
        "chunk_size": tune.choice([1024, 2048, 4096, 8192, 16384, 32768, 65536]),
        "overlap": tune.choice([256, 512, 768]),
    }


def hp_space_malconvgct(trial: Any) -> dict[str, float | int]:  # pylint: disable=unused-argument
    return {
        "stride": tune.choice([64, 128, 192, 256, 320, 384, 448, 512]),
        "window_size": tune.choice([64, 128, 192, 256, 320, 384, 448, 512]),
        "channels": tune.choice([64, 128, 192]),
        "chunk_size": tune.choice([1024, 2048, 4096, 8192, 16384, 32768, 65536]),
        "overlap": tune.choice([256, 512, 768]),
    }


def hp_space_mymalconv(trial: Any) -> dict[str, float | int]:  # pylint: disable=unused-argument
    return {
        "stride": tune.choice([64, 128, 192, 256, 320, 384, 448, 512]),
        "window_size": tune.choice([64, 128, 192, 256, 320, 384, 448, 512]),
        "channels": tune.choice([64, 128, 192]),
        "hidden_size": tune.choice([128, 256, 512, 768, 1024]),
    }

