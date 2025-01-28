"""
Prettier printing of some hugginface things.
"""

from pprint import pformat

from transformers import (
    DefaultDataCollator,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
)

from src.utils import count_parameters


def print_tokenizer(tokenizer: PreTrainedTokenizerFast) -> None:
    keys = [
        "name_or_path",
        "vocab_size",
        "model_max_length",
        "is_fast",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
        "model_input_names",
    ]

    tokens_to_ids = {k: v for k, v in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)}  # pylint: disable=unnecessary-comprehension
    specials = [(k, v, tokens_to_ids[v]) for k, v in tokenizer.special_tokens_map.items()]

    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    for k in keys:
        if hasattr(tokenizer, k):
            print(f"\t{k}: {getattr(tokenizer, k)}")
    print(f"\tspecials: {pformat(specials)}")


def print_data_collator(data_collator: DefaultDataCollator) -> None:
    keys = [
        "padding",
        "max_length",
        "pad_to_multiple_of",
        "return_tensors",
        "mlm",
        "mlm_probability",
    ]

    print(f"DataCollator: {data_collator.__class__.__name__}")
    if hasattr(data_collator, "tokenizer") and data_collator.tokenizer is not None:
        print(f"\ttokenizer: {data_collator.tokenizer.__class__.__name__}")
    for k in keys:
        if hasattr(data_collator, k):
            print(f"\t{k}: {getattr(data_collator, k)}")


def print_config(config: PretrainedConfig) -> None:

    def simplify_ids_and_labels(s: str, key: str) -> str:
        idx = s.find(key)
        if idx == -1:  # Config must be for langauge modeling.
            return s
        idx += len(key)  # index of either "{" or "N" character
        if s[idx] == "{":  # Strip
            idx_end = s.index("}", idx, None)  # index of closing "}"
            s = s[:idx + 1] + "..." + s[idx_end:]
        return s

    print(f"Config: {config.__class__.__name__}")
    s = pformat(config)
    s = simplify_ids_and_labels(s, '"id2label": ')
    s = simplify_ids_and_labels(s, '"label2id": ')
    print(s)


def print_model(model: PreTrainedModel) -> None:
    print(f"Model: {model.__class__.__name__}")
    print(model)
    print("\tNumber of parameters:")
    print(f"\t\tTotal:     {count_parameters(model)}")
    try:
        c_embd = count_parameters(model.backbone.embeddings)
        c_back = count_parameters(model.backbone) - c_embd
        for h in ["head_clm", "head_mlm", "head_clf"]:
            if hasattr(model, h):
                c_head = count_parameters(getattr(model, h))
                break
        else:
            raise AttributeError()
        print(f"\t\tEmbedding: {c_embd}")
        print(f"\t\tBackbone:  {c_back}")
        print(f"\t\tHead:      {c_head}")
    except AttributeError:
        pass

    try:
        inp = model.get_input_embeddings()
        out = model.get_output_embeddings()
        print(f"\tEmbedding weights the same: {inp.weight is out.weight}")
        print(f"\tEmbedding biases the same:  {inp.bias is out.bias}")
    except (AttributeError, NotImplementedError):
        pass
