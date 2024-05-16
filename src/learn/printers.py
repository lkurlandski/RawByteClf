"""
Prettier printing of some hugginface things.
"""

from pprint import pformat

from transformers import (
    DefaultDataCollator,
    PreTrainedTokenizerFast,
    PretrainedConfig,
)


def print_tokenizer(tokenizer: PreTrainedTokenizerFast) -> None:
    keys = [
        "name_or_path",
        "vocab_size",
        "model_max_length",
        "is_fast",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
    ]

    tokens_to_ids = {k: v for k, v in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)}
    specials = [(k, v, tokens_to_ids[v]) for k, v in tokenizer.special_tokens_map.items()]

    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    for k in keys:
        print(f"\t{k}: {getattr(tokenizer, k)}")
    print(f"\tmodel_input_names: {tokenizer.model_input_names}")
    print(f"\tspecials: {pformat(specials)}")


def print_data_collator(data_collator: DefaultDataCollator) -> None:
    keys = ["padding", "max_length", "pad_to_multiple_of", "return_tensors"]
    print(f"DataCollator: {data_collator.__class__.__name__}")

    if data_collator.tokenizer is not None:
        print(f"\ttokenizer: {data_collator.tokenizer.__class__.__name__}")
    for k in keys:
        print(f"\t{k}: {getattr(data_collator, k)}")


def print_config(config: PretrainedConfig) -> None:
    print(f"Config: {config.__class__.__name__}")
    s = pformat(config)

    key = '"id2label": '
    idx = s.index(key) + len(key)  # index of either "{" or "N" character
    if s[idx] == "{":  # Strip
        idx_end = s.index("}", idx, None)  # index of closing "}"
        s = s[:idx + 1] + "..." + s[idx_end:]

    key = '"label2id": '
    idx = s.index(key) + len(key)  # index of either "{" or "N" character
    if s[idx] == "{":  # Strip
        idx_end = s.index("}", idx, None)  # index of closing "}"
        s = s[:idx + 1] + "..." + s[idx_end:]

    print(s)
