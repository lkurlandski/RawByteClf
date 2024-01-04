"""
Test HRRFormer implementation and compare to BERT.
"""

from argparse import ArgumentParser
from datetime import datetime
from functools import partial
import os
import sys

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    BertConfig,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    BertForMaskedLM,
    BertLMHeadModel as BertForCausalLM,
    BertForSequenceClassification,
    BertModel,
)


from src.hrrformer import (
    HRRConfig,
    HRRModel,
    HRRForCLM,
    HRRForMLM,
    HRRForSequenceClassification,
)


NUM_HIDDEN_LAYERS = 1
NUM_TRAIN_EPOCHS = 1
MAX_POSITION_EMBEDDINGS = 128
BLOCK_SIZE = 128
NUM_PROC = 4
BATCH_SIZE = 128
PAD_TO_MULTIPLE_OF = 8


def group_texts(examples, block_size: int):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def get_dataset(task: str, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
    if task == "clm" or task == "mlm":
        dataset = load_dataset("eli5", split="train_asks")
        dataset = dataset.train_test_split(test_size=0.1)
        dataset = dataset.flatten()
        dataset = dataset.map(
            lambda examples: tokenizer(
                [" ".join(x) for x in examples["answers.text"]],
                truncation=True,
                max_length=MAX_POSITION_EMBEDDINGS,
            ),
            batched=True,
            num_proc=NUM_PROC,
            remove_columns=dataset["train"].column_names,
        )
        dataset = dataset.map(
            partial(group_texts, block_size=BLOCK_SIZE),
            batched=True,
            num_proc=NUM_PROC,
        )
        return dataset

    elif task == "clf":
        dataset = load_dataset("ag_news")
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=MAX_POSITION_EMBEDDINGS,
            ),
            batched=True,
            num_proc=NUM_PROC,
        )
        dataset = dataset.rename_column("label", "labels")
        return dataset

    raise RuntimeError(f"Unknown task: {task}")


def get_config(
    task: str, model: str, tokenizer: PreTrainedTokenizerFast, dataset: DatasetDict
) -> BertConfig | HRRConfig:
    kwds = {
        "vocab_size": len(tokenizer),
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
    }
    if task == "clf":
        kwds.update({
            "num_labels": dataset["train"].info.features["labels"].num_classes,
            "id2label": {i: l for i, l in enumerate(dataset["train"].info.features["labels"].names)},
            "label2id": {l: i for i, l in enumerate(dataset["train"].info.features["labels"].names)},
        })

    if model == "bert":
        return BertConfig(**kwds)
    elif model == "hrr":
        return HRRConfig(**kwds)
    
    raise RuntimeError(f"Unknown model: {model}")


def get_model(task: str, model: str, config: BertConfig | HRRConfig):
    if task == "clm":
        if model == "bert":
            return BertForCausalLM(config)
        elif model == "hrr":
            return HRRForCLM(config)
    elif task == "mlm":
        if model == "bert":
            return BertForMaskedLM(config)
        elif model == "hrr":
            return HRRForMLM(config)
    elif task == "clf":
        if model == "bert":
            return BertForSequenceClassification(config)
        elif model == "hrr":
            return HRRForSequenceClassification(config)

    raise RuntimeError(f"Unknown task: {task} or unknown model: {model}")


def get_data_collator(task: str, tokenizer: PreTrainedTokenizerFast):
    if task == "clf":
        return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=PAD_TO_MULTIPLE_OF)
    elif task == "mlm":
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=PAD_TO_MULTIPLE_OF)
    elif task == "clf":
        return DataCollatorWithPadding(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=PAD_TO_MULTIPLE_OF)

    raise RuntimeError(f"Unknown task: {task}")


def get_training_arguments(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-5,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        debug="underflow_overflow",
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--task", type=str, choices=["clm", "mlm", "clf"], default="clf")
    parser.add_argument("--model", type=str, choices=["bert", "hrr"], default="bert")
    args = parser.parse_args()

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-cased")
    print(f"{tokenizer=}")

    dataset: DatasetDict = get_dataset(args.task, tokenizer)
    print(f"{dataset=}")

    config: BertConfig | HRRConfig = get_config(args.task, args.model, tokenizer, dataset)
    print(f"{config=}")

    model: BertModel | HRRModel = get_model(args.task, args.model, config)
    print(f"{model=}")

    data_collator: DataCollatorForLanguageModeling | DataCollatorWithPadding = get_data_collator(args.task, tokenizer)

    trainer = Trainer(
        model=model,
        args=get_training_arguments(f"./tmp/{args.task}/{args.model}"),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )
    trainer.train()
