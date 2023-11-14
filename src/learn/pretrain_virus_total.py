"""
Pretrain on unlabeled PE files and finetune on BODMAS dataset.
"""

from collections import OrderedDict
import os
from pathlib import Path
from pprint import pprint
import sys
import time
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# pylint: disable=wrong-import-position

from datasets import Dataset, IterableDataset, IterableDatasetDict, concatenate_datasets
from datasets.formatting.formatting import LazyBatch, LazyRow
import numpy as np
from tokenizers import models, pre_tokenizers, Tokenizer, Regex
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    BertConfig,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    LongformerConfig,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from cfg import INPUT_PATH, OUTPUT_PATH
from utils import count_parameters


# torch.backends.cudnn.enabled = False


NUM_ITERABLE_DATASET_SHARDS = 16
MAX_LENGTH = 2 ** 12
SPECIALS = OrderedDict({
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "mask_token": "<msk>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "cls_token": "<cls>",
    "sep_token": "<sep>",
})


def preprocess_a(examples: Any) -> dict:
    """This is about half the speed of preprocess_b, but lets us use the vast HF ecosystem."""
    return {
        "text": [b.decode("latin1") for b in examples["bytes"]],
    }


def preprocess_b(examples: Any) -> dict:
    return {
        "input_ids": [np.frombuffer(b, dtype=np.uint8) for b in examples["bytes"]],
    }


dataset = concatenate_datasets([
    #Dataset.load_from_disk(INPUT_PATH / "virus_total_exe"),
    Dataset.load_from_disk(INPUT_PATH / "virus_total_dll"),
]).select(range(256))
dataset.cleanup_cache_files()
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"{dataset}")
NUM_SAMPLES = {"train": len(dataset["train"]), "test": len(dataset["test"])}
# dataset = IterableDatasetDict({
#     "train": dataset["train"].to_iterable_dataset(num_shards=NUM_ITERABLE_DATASET_SHARDS),
#     "test": dataset["test"].to_iterable_dataset(num_shards=NUM_ITERABLE_DATASET_SHARDS)
# })
# print(f"{dataset}")



def tokenize_fn(examples: Any) -> dict:
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, pad_to_multiple_of=8,
)

attention_window = 512
config = LongformerConfig(
    attention_window=attention_window,
    sep_token_id=tokenizer.sep_token_id,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_hidden_layers=1,
    num_attention_heads=12,
    vocab_size=264,
    hidden_size=768,
    intermediate_size=3072,
    max_position_embeddings=attention_window + MAX_LENGTH,
)
model = AutoModelForMaskedLM.from_config(config)
print(f"{round(count_parameters(model) / 1e6, 2)}M parameters.")

training_arguments = TrainingArguments(
    output_dir=(OUTPUT_PATH / "tmp").as_posix(),
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    save_total_limit=5,
    fp16=True,
    dataloader_num_workers=1,
    group_by_length=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    #max_steps=NUM_SAMPLES["train"] // 8,
)

callbacks = []
compute_metrics = None

print("Preprocessing...")
dataset = dataset.map(preprocess_a, batched=True, batch_size=1024, remove_columns=["bytes"])
print("Tokenizing...")
dataset = dataset.map(tokenize_fn, batched=True, batch_size=1024, remove_columns=["text", "labels"])

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=callbacks,
    compute_metrics=compute_metrics,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

trainer.train()

#finetune_dataset = Dataset.load_from_disk(INPUT_PATH / "bodmas")



