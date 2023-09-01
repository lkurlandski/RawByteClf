"""
Train and evaluate the models for malware family classification.
"""

from datetime import datetime
import json
from pprint import pformat, pprint
import os
import sys

from datasets import concatenate_datasets, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from cfg import BR
from helpers import OutputHelper
from train import get_config_hf, ModelArgs, CallbackArgs
from tokenization import get_fast_tokenizer
from utils import count_parameters


def main(model_args: ModelArgs, callback_args: CallbackArgs, training_args: TrainingArguments):
    if model_args.task == "mlm":
        MLM = True
        AutoModelForLM: type = AutoModelForMaskedLM
    elif model_args.task == "clm":
        MLM = False
        AutoModelForLM: type = AutoModelForCausalLM
    else:
        raise ValueError(f"{model_args.task=} not supported.")

    # We want to get both the labeled and unlabeled datasets and concatenate them for pretraining.
    # So we need to create two OutputHelper objects, one for the pretraining task and one for the
    # classification task. We can use the same model arguments for both, except for the task
    oh, oh_clf = [
        OutputHelper(
            algorithm=model_args.algorithm,
            vocab_size=model_args.vocab_size,
            num_tok=model_args.num_tok,
            max_length=model_args.max_length,
            num=model_args.num,
            task=task,
            model=model_args.model,
            scale=model_args.scale,
            pretrain_task=model_args.pretrain_task,
        )
        for task in (model_args.task, "clf")
    ]
    print(f"{oh=}")

    tokenizer = get_fast_tokenizer(oh.tokenizer_file, model_args.max_length)
    print(f"{tokenizer=}")
    print(BR, flush=True)

    d_1 = DatasetDict.load_from_disk(oh.dataset_dir)
    d_2 = DatasetDict.load_from_disk(oh_clf.dataset_dir)
    dataset = DatasetDict(
        {
            "tr": concatenate_datasets([d_1["tr"], d_2["tr"]]),
            "vl": concatenate_datasets([d_1["vl"], d_2["vl"]]),
            "ts": concatenate_datasets([d_1["ts"], d_2["ts"]]),
        }
    ).remove_columns("label")
    print(f"{dataset=}")
    print(BR, flush=True)

    config = get_config_hf(
        model_args.model,
        tokenizer,
        model_args.max_length,
        model_args.scale,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=MLM)
    callbacks = []
    if callback_args.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            callback_args.early_stopping_patience,
            callback_args.early_stopping_threshold,
        )
        callbacks.append(early_stopping_callback)

    model = AutoModelForLM.from_config(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["tr"],
        eval_dataset=dataset["vl"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    print(f"{config=}")
    print(f"{model=}")
    print(f"{data_collator=}")
    print(f"{callbacks=}")
    print(f"{count_parameters(model)=}")
    print(BR, flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    oh.mkdir(exist_ok=True)

    if training_args.do_train:
        # TrainingArguments are immutable in newest transformers version
        object.__setattr__(training_args, "output_dir", oh.checkpoints_dir.as_posix())
        trainer.train(training_args.resume_from_checkpoint)
        if training_args.load_best_model_at_end:
            model.save_pretrained(oh.best_model_dir.as_posix())
        with open(oh.log_history_path, "w") as fp:
            json.dump(trainer.state.log_history, fp, indent=4)

    if training_args.do_eval:
        model = AutoModelForLM.from_pretrained(oh.best_model_dir.as_posix())
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
        results = trainer.evaluate(dataset["ts"])
        with open(oh.test_results_path, "w") as fp:
            json.dump(results, fp, indent=4)


def cli():
    parser = HfArgumentParser((ModelArgs, CallbackArgs, TrainingArguments))
    model_args, callback_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.dataloader_num_workers and training_args.dataloader_num_workers < 0:
        training_args.dataloader_num_workers = int(
            len(os.sched_getaffinity(0)) // abs(training_args.dataloader_num_workers)
        )
    assert training_args.load_best_model_at_end

    print(f"model_args={pformat(model_args)}")
    print(f"callback_args={pformat(callback_args)}")
    print(f"training_args={pformat(training_args)}")
    print(BR, flush=True)
    main(model_args, callback_args, training_args)
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)


def debug() -> None:
    pass


if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{BR}", flush=True)
    if len(sys.argv) == 1 or sys.argv[1] == "--debug":
        debug()
    else:
        cli()
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)
