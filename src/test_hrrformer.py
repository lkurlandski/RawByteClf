"""
Test HRRFormer implementation and compare to BERT.
"""

from argparse import ArgumentParser
from datetime import datetime
from functools import partial
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset, Dataset, DatasetDict
import evaluate
import numpy as np
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
    HRRLMHeadModel as HRRForCausalLM,
    HRRForMaskedLM,
    HRRForSequenceClassification,
)
from src.mamba import (
    ModelArgs as MambaConfig,
    Mamba as MambaForCausalLM,
    # MambaConfig,
    # MambaModel,
    # MambaLMHeadModel as MambaForCausalLM,
    # MambaForMaskedLM,
    # MambaForSequenceClassification,
)


HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 1
NUM_TRAIN_EPOCHS = 10
MAX_POSITION_EMBEDDINGS: int = None  # clf: 512, mlm: 512, clm: 128
BATCH_SIZE: int = None # clf: 256, mlm: 64, clm: 64
PAD_TO_MULTIPLE_OF = 8
NUM_PROC = 4


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
            partial(group_texts, block_size=MAX_POSITION_EMBEDDINGS),
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
) -> BertConfig | HRRConfig | MambaConfig:
    kwds = {
        "vocab_size": len(tokenizer),
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "hidden_size": HIDDEN_SIZE,
    }
    if task == "clf":
        kwds.update({
            "num_labels": dataset["train"].info.features["labels"].num_classes,
            "id2label": {i: l for i, l in enumerate(dataset["train"].info.features["labels"].names)},
            "label2id": {l: i for i, l in enumerate(dataset["train"].info.features["labels"].names)},
        })
    if task == "clm":
        kwds.update({
            "is_decoder": True,
            # "add_cross_attention": True,  # only for seq2seq
        })

    if model == "bert":
        return BertConfig(**kwds)
    elif model == "hrr":
        return HRRConfig(**kwds)
    elif model == "mamba":
        return MambaConfig(
            d_model=768,
            n_layer=12,
            vocab_size=len(tokenizer),
        )

    raise RuntimeError(f"Unknown model: {model}")


def get_model(task: str, model: str, config: BertConfig | HRRConfig):
    if task == "clm":
        if model == "bert":
            return BertForCausalLM(config)
        elif model == "hrr":
            return HRRForCausalLM(config)
        elif model == "mamba":
            return MambaForCausalLM(config)
    elif task == "mlm":
        if model == "bert":
            return BertForMaskedLM(config)
        elif model == "hrr":
            return HRRForMaskedLM(config)
        elif model == "mamba":
            return MambaForMaskedLM(config)
    elif task == "clf":
        if model == "bert":
            return BertForSequenceClassification(config)
        elif model == "hrr":
            return HRRForSequenceClassification(config)
        elif model == "mamba":
            return MambaForSequenceClassification(config)

    raise RuntimeError(f"Unknown task: {task} or unknown model: {model}")


def get_data_collator(task: str, tokenizer: PreTrainedTokenizerFast):
    if task == "clf":
        return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=PAD_TO_MULTIPLE_OF)
    elif task == "mlm":
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=PAD_TO_MULTIPLE_OF)
    elif task == "clm":
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=PAD_TO_MULTIPLE_OF)

    raise RuntimeError(f"Unknown task: {task}")


def get_compute_metrics(task: str):

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    if task == "clf":
        return compute_metrics
    elif task == "mlm" or task == "clm":
        return None

    raise RuntimeError(f"Unknown task: {task}")


def get_training_arguments(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=250,
        eval_steps=250,
        logging_steps=50,
        learning_rate=2e-5,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_total_limit=3,
        # debug="underflow_overflow",
        # use_cpu=True,
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--task", type=str, choices=["clm", "mlm", "clf"], default="clm")
    parser.add_argument("--model", type=str, choices=["bert", "hrr", "mamba"], default="hrr")
    parser.add_argument("--max_position_embeddings", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    MAX_POSITION_EMBEDDINGS = args.max_position_embeddings
    BATCH_SIZE = args.batch_size

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-cased")
    if args.model == "mamba":
        tokenizer.model_input_names.remove("attention_mask")
    print(f"{tokenizer=}")
    print(f"{tokenizer.model_input_names=}")

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
        compute_metrics=get_compute_metrics(args.task),
    )

    print(f"TRAINING @{datetime.now()}\n{'-' * 88}", flush=True)
    trainer.train()
    print(f"FINISHED @{datetime.now()}\n{'-' * 88}", flush=True)
