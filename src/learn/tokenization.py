"""
Handles preprocessing of malware bytes.
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")
# pylint: enable=wrong-import-position

from collections import OrderedDict
from typing import Optional

from tokenizers import models, pre_tokenizers, processors, Tokenizer, Regex
from transformers import PreTrainedTokenizerFast


SPECIALS = OrderedDict(
    {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "mask_token": "<msk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "cls_token": "<cls>",
        "sep_token": "<sep>",
    }
)
SPECIALS_IDS = {k: i for i, k in enumerate(SPECIALS)}


class FastTokenizerForModelsThatRequireCLSToken(PreTrainedTokenizerFast):
    def build_inputs_with_special_tokens(self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
    ) -> list:
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]
        return output


def get_tokenizer_object(model_requires_cls_token: bool) -> Tokenizer:
    alphabet = [bytes([i]).decode("latin1") for i in range(256)]
    vocab = {v: i for i, v in enumerate(SPECIALS.values())} | {
        v: i for i, v in enumerate(alphabet, start=len(SPECIALS))
    }

    model = models.WordLevel(vocab=vocab, unk_token=SPECIALS["unk_token"])
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            # For some reason, when \n\n is encountered, the Regex(".") fails to
            # split, so we need to split on \n first then split on . (matches all).
            pre_tokenizers.Split(Regex("\n"), behavior="isolated"),
            pre_tokenizers.Split(Regex("."), behavior="isolated"),
        ]
    )

    if model_requires_cls_token:
        tokenizer.post_processor = processors.BertProcessing(
            sep=(SPECIALS["sep_token"], SPECIALS_IDS["sep_token"]),
            cls=(SPECIALS["cls_token"], SPECIALS_IDS["cls_token"]),
        )

    return tokenizer


def get_tokenizer(model_requires_cls_token: bool, **kwds) -> PreTrainedTokenizerFast:
    """
    kwds
       model_max_length: will caused the tokenizer to trim the tokenized input.
    """
    tokenizer = get_tokenizer_object(model_requires_cls_token)

    if model_requires_cls_token:
        tokenizer = FastTokenizerForModelsThatRequireCLSToken(tokenizer_object=tokenizer, **(kwds | SPECIALS))
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **(kwds | SPECIALS))

    tokenizer.add_special_tokens(SPECIALS)  # TODO: is this line necessary?
    return tokenizer


if __name__ == "__main__":
    t = get_tokenizer(model_requires_cls_token=True, model_max_length=65536)
    print(f"{t}\n{'-' * 80}")
    t = get_tokenizer(model_requires_cls_token=False, model_max_length=65536)
    print(f"{t}\n{'-' * 80}")
