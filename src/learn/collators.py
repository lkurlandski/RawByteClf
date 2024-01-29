"""
Group examples together.

TODO: inserting the CLS token should take place here?
"""

import math
import os
import sys
from typing import Any, Literal

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch import BoolTensor, LongTensor, tensor
from torch.nn.utils.rnn import pad_sequence


class DataCollator:

    def __init__(
        self,
        pad_token_id: int,
        special_ids: tuple[int] = tuple(),
        pad_to_multiple_of: int = 1,
        return_token_type_ids: bool = False,
        return_attention_mask: bool = False,
        return_special_tokens_mask: bool = False,
    ) -> None:
        self.pad_to_multiple_of = pad_to_multiple_of
        self.special_ids = special_ids
        self.return_token_type_ids = return_token_type_ids
        self.return_attention_mask = return_attention_mask
        self.return_special_tokens_mask = return_special_tokens_mask
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[Literal["input_ids", "labels"], LongTensor]]) -> dict[str, LongTensor]:
        # Pad the longest sequence to a multiple of self.pad_to_multiple_of if necessary.
        if self.pad_to_multiple_of is not None:
            lengths = [len(e["input_ids"]) for e in examples]
            max_idx = torch.argmax(tensor(lengths))
            max_val = len(examples[max_idx]["input_ids"])
            length = math.ceil(max_val / self.pad_to_multiple_of * self.pad_to_multiple_of)
            if (padding := length - max_val) > 0:
                pad_tensor = torch.full(padding, self.pad_token_id, dtype=torch.long)
                examples["input_ids"][max_idx] = torch.cat([examples["input_ids"], pad_tensor])

        # Pad all sequences to the same length.
        batch = {
            "input_ids": pad_sequence(
                [e["input_ids"] for e in examples],
                batch_first=True,
                padding_value=self.pad_token_id,
            )
        }
        if "labels" in examples[0]:
            batch["labels"] = torch.stack([e["labels"] for e in examples])
        if self.return_attention_mask:
            batch["attention_mask"] = self.get_special_tokens_mask(
                batch["input_ids"], (self.pad_token_id,),
            )

        return batch

    @staticmethod
    def get_special_tokens_mask(x: LongTensor, specials: list[int]) -> BoolTensor:
        """
        Mask is True (1) if token is special, else False (0).
        """
        m = torch.any(
            torch.eq(x.unsqueeze(-1), torch.tensor(specials).unsqueeze(0).unsqueeze(0)),
            dim=-1,
        ).bool()
        assert m.shape == x.shape
        return m


class DataCollatorForCLM(DataCollator):

    def __call__(self, examples: list[dict[Literal["input_ids"], LongTensor]]) -> Any:
        batch = super().__call__(examples)
        labels: LongTensor = batch["input_ids"].clone()
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels
        return batch


class DataCollatorForMLM(DataCollator):

    def __init__(
        self,
        pad_token_id: int,
        specials: list[int],
        mask_token_id: int,
        vocab_size: int,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: int = 1,
        return_token_type_ids: bool = False,
        return_attention_mask: bool = False,
        return_special_tokens_mask: bool = False,
    ) -> None:
        super().__init__(
            pad_token_id,
            specials,
            pad_to_multiple_of,
            return_token_type_ids,
            return_attention_mask,
            return_special_tokens_mask,
        )
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability

    def __call__(self, examples: list[dict[Literal["input_ids"], LongTensor]]) -> Any:
        batch = super().__call__(examples)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"],
            special_tokens_mask=self.get_special_tokens_mask(batch["input_ids"], self.special_ids),
        )
        return batch

    def mask_tokens(
        self, inputs: LongTensor, special_tokens_mask: BoolTensor = None
    ) -> tuple[LongTensor, LongTensor]:
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def test() -> None:
    from datasets import DatasetDict
    from transformers import (
        BertTokenizerFast,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        PretrainedConfig,
        DataCollatorWithPadding,
        DataCollatorForLanguageModeling,
        Trainer,
    )

    from src.architectures.test import (
        get_dataset, get_config, get_model, get_compute_metrics, get_training_arguments
    )

    task = "clf"

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-cased")
    print(f"{tokenizer=}")
    print(f"{tokenizer.model_input_names=}")
    print(f"{tokenizer.all_special_ids=}")
    print(f"{tokenizer.all_special_tokens=}")

    dataset: DatasetDict = get_dataset(task, tokenizer)
    print(f"{dataset=}")

    config: PretrainedConfig = get_config(task, "bert", tokenizer, dataset)
    print(f"{config=}")

    model: PreTrainedModel = get_model(task, "bert", config)
    print(f"{model=}")

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    data_collator = DataCollator(
        tokenizer.pad_token_id,
        special_ids=tokenizer.all_special_ids,
        pad_to_multiple_of=8,
        return_attention_mask=True,
    )

    trainer = Trainer(
        model=model,
        args=get_training_arguments(f"./tmp/collators/hf"),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(task),
    )

    trainer.train()


if __name__ == "__main__":
    test()
