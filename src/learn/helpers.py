"""
Utilies to help with the output of the training process.
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

from dataclasses import dataclass, field
from datetime import datetime
import errno
import json
import os
from pathlib import Path
from pprint import pprint
import shutil
import sys
from typing import Any, Hashable, Optional

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.cfg import OUTPUT_PATH
from src.utils import get_highest_path, is_jsonable
from src.learn.utils import str_or_bool_to_str, float_to_int


@dataclass
class Args:

    model_name_or_path: str = field()
    max_length: int = field()
    task: str = field()
    depth: int = field(default=1)
    streaming: bool = field(default=False)
    exit_after_map: bool = field(default=False)
    ft_freeze_positional_embeddings: bool = field(default=False)
    ft_duplicate_positional_embeddings: bool = field(default=False)
    ft_initialize_positional_embeddings: bool = field(default=False)
    root: Path = field(default=OUTPUT_PATH)
    do_tune: bool = field(default=False)
    bodmas_min_freq: Optional[int] = field(default=None)
    bodmas_top_k: Optional[int] = field(default=None)
    arch_config_file: Optional[Path] = field(
        default=None,
        metadata={"help": "Location of a configuration file to use for the architecture."},
    )
    arch_config: Optional[dict | str] = field(
        default=None,
        metadata={"help": "Configuration dict to use for the architecture. Mutally exclusive with arch_config_file."},
    )
    subset: Optional[int] = field(default=None)
    tr_size: float = field(default=0.8, metadata={"help": "If > 1, then it is the number of samples."})
    vl_size: float = field(default=0.1, metadata={"help": "If > 1, then it is the number of samples."})
    ts_size: float = field(default=0.1, metadata={"help": "If > 1, then it is the number of samples."})
    skip_eval_check: bool = field(default=False)
    auto_find_batch_size_and_gradient_accumulation_steps: bool = field(default=False)
    early_stopping: bool = field(default=False)
    early_stopping_patience: int = field(default=1)
    early_stopping_threshold: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.ft_freeze_positional_embeddings = str_or_bool_to_str(self.ft_freeze_positional_embeddings)
        self.ft_duplicate_positional_embeddings = str_or_bool_to_str(self.ft_duplicate_positional_embeddings)
        self.ft_initialize_positional_embeddings = str_or_bool_to_str(self.ft_initialize_positional_embeddings)
        self.streaming = str_or_bool_to_str(self.streaming)
        self.exit_after_map = str_or_bool_to_str(self.exit_after_map)
        self.do_tune = str_or_bool_to_str(self.do_tune)
        self.skip_eval_check = str_or_bool_to_str(self.skip_eval_check)
        self.auto_find_batch_size_and_gradient_accumulation_steps = str_or_bool_to_str(self.auto_find_batch_size_and_gradient_accumulation_steps)

        if self.arch_config_file and self.arch_config:
            raise ValueError("Cannot specify both arch_config_file and arch_config.")
        if self.arch_config and isinstance(self.arch_config, str):
            self.arch_config = json.loads(self.arch_config)
        if self.arch_config_file:
            with open(self.arch_config_file) as fp:
                self.arch_config = json.load(fp)

        self.tr_size = float_to_int(self.tr_size) if self.tr_size > 1 else self.tr_size
        self.vl_size = float_to_int(self.vl_size) if self.vl_size > 1 else self.vl_size
        self.ts_size = float_to_int(self.ts_size) if self.ts_size > 1 else self.ts_size


class OutputHelper:

    # The values from the TrainingArguments thats will be hashed.
    trainer_config_relevant_keys = [
        # Optimizer
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "learning_rate",
        "lr_scheduler_type",
        "max_grad_norm",
        "optim",
        "warmup_ratio",
        "weight_decay",
        # Batch size
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        # Numeric types
        "tf32",
        "fp16",
        "bf16",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int,
        task: str,
        tr_size: int | float,
        depth: int,
        bodmas_min_freq: Optional[int],
        bodmas_top_k: Optional[int],
        ft_freeze_positional_embeddings: bool | str,
        ft_duplicate_positional_embeddings: bool | str,
        ft_initialize_positional_embeddings: bool | str,
        root: Path,
        arch_config: Optional[dict] = None,
        trainer_config: Optional[dict] = None,
    ) -> None:
        """
        {max_length}/{task}/{tr_size}/{
         {depth} |
         {min_freq}/{top_k}/{freeze/{duplicate}/{initialize}
        }/
        """

        self.root = Path(root)

        args = [
            str(max_length),
            task,
            str(tr_size),
        ]
        if task == "clf":
            args.extend(
                [
                    str(bodmas_min_freq),
                    str(bodmas_top_k),
                    str(str_or_bool_to_str(ft_freeze_positional_embeddings)),
                    str(str_or_bool_to_str(ft_duplicate_positional_embeddings)),
                    str(str_or_bool_to_str(ft_initialize_positional_embeddings)),
                ]
            )
        elif task in ("mlm", "clm"):
            args.extend(
                [
                    str(depth),
                ]
            )
        self.arch_config = arch_config
        self.trainer_config = trainer_config

        # If model_name_or_path is a path and it exists, then we're finetuning something.
        # In this case the model_name_or_path is a path and it already contains the model name,
        # so we don't need to include it in basepath Otherwise, its the firt part of basepath.
        self.finetuning = Path(model_name_or_path).exists()
        if self.finetuning:
            self.basepath = Path(model_name_or_path).joinpath(*args)
        else:
            args.insert(0, model_name_or_path)
            self.basepath = self.root.joinpath(*args)

    def __repr__(self) -> str:
        return self.path.as_posix()

    def __str__(self) -> str:
        return self.path.as_posix()

    @property
    def path(self) -> Path:
        """
        If finetuning, we want to place the finetuned model within the checkpoint its being
        finetuned from. In this circumstance, we can assume that the architecture is the same,
        so we don't include that in the path construction. Otherwise, we assume we're training a
        model from scratch and we want the model_name_or_path as well as its architectural details.
        """
        if self.finetuning:
            tail = self.trainer_path()
        else:
            tail = self.arch_path() / self.trainer_path()

        return self.basepath / tail

    @property
    def best_model_dir(self) -> Path:
        with open(self.last_checkpoint / "trainer_state.json") as fp:
            state = json.load(fp)
        best_model_checkpoint = state["best_model_checkpoint"]
        return Path(best_model_checkpoint)

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def config_file(self) -> Path:
        return self.path / "config.json"

    @property
    def arch_config_file(self) -> Path:
        return self.path / "arch_config.json"

    @property
    def trainer_config_file(self) -> Path:
        return self.path / "trainer_config.json"

    @property
    def test_results_dir(self) -> Path:
        return self.path / "test_results"

    @property
    def test_results_file(self) -> Path:
        return self.test_results_dir / "results.json"

    @property
    def initial_validation_results_file(self) -> Path:
        return self.path / "initial_validation_results.json"

    @property
    def trainer_output_file(self) -> Path:
        return self.path / "trainer_output_file.json"

    @property
    def test_predictions_file(self) -> Path:
        return self.test_results_dir / "predictions.txt"

    @property
    def test_probas_file(self) -> Path:
        return self.test_results_dir / "probas.txt"

    @property
    def test_labels_file(self) -> Path:
        return self.test_results_dir / "labels.txt"

    @property
    def test_confusion_matrix_file(self) -> Path:
        return self.test_results_dir / "confusion_matrix.png"

    @property
    def tuning_results_dir(self) -> Path:
        return self.path / "tuning_results"

    @property
    def last_checkpoint(self) -> Path:
        return get_highest_path(self.checkpoints_dir, lstrip="checkpoint-")

    @property
    def tensor_log_path(self) -> Path:
        return self.path / "tensor_log_path"

    def mkdir(self) -> None:
        self.path.mkdir(exist_ok=True, parents=True)

        # Dump the architecture configuration to a file.
        if self.arch_config:
            d = {k: v if is_jsonable(v) else str(v) for k, v in self.arch_config.items()}
            with open(self.arch_config_file, "w") as fp:
                json.dump(d, fp, indent=4)

        # Dump the trainer configuration to a file.
        if self.trainer_config:
            d = {k: v if is_jsonable(v) else str(v) for k, v in self.trainer_config.items()}
            with open(self.trainer_config_file, "w") as fp:
                json.dump(d, fp, indent=4)

    def rmdir(self, ignore_config: bool = False, force: bool = False) -> None:
        """
        Remove the output directory and all its contents. Default behaviour is to only remove the
        directory if it is empty. If ignore_config is True, then will delete the directory if the
        only two files found within it are the arch_config and trainer_config files. If force is
        True, then will hard delete the directory and all its contents.
        """
        if not self.path.exists():
            return

        if force:
            shutil.rmtree(self.path)
            return

        files = list(filter(lambda p: p.is_file(), self.path.glob("*")))
        if len(files) == 0:
            shutil.rmtree(self.path)
            return

        if ignore_config and len(files) == 2:
            if {self.arch_config_file, self.trainer_config_file} == set(files):
                shutil.rmtree(self.path)
                return

        raise OSError(errno.ENOTEMPTY, os.strerror(errno.ENOTEMPTY), self.path)

    def arch_path(self, arch_config: Optional[dict[str, Any]] = None) -> Path:
        arch_config = arch_config if arch_config is not None else self.arch_config
        if arch_config is None:
            return Path()
        return Path().joinpath(*[f"{k}--{v}" for k, v in arch_config.items()])

    def trainer_path(self, trainer_config: Optional[dict[str, Any]] = None) -> Path:
        trainer_config = trainer_config if trainer_config is not None else self.trainer_config
        if trainer_config is None:
            return Path()
        d = {k: v for k, v in trainer_config.items() if k in self.trainer_config_relevant_keys}
        return Path().joinpath(*[f"{k}--{v}" for k, v in d.items()])

    # @property
    # def arch_config_hash(self) -> str:
    #     raise DeprecationWarning("Hashing the arch config is depricated.")
    #     if not self.arch_config:
    #         return ""
    #     return self.get_hash(self.arch_config)

    # @property
    # def trainer_config_hash(self) -> str:
    #     if not self.trainer_config:
    #         return ""
    #     d = {k: v for k, v in self.trainer_config.items() if k in self.trainer_config_relevant_keys}
    #     return self.get_hash(d)

    # @staticmethod
    # def get_hash(d: dict[str, Any]) -> str:
    #     """
    #     Tries to convert not hashable values to hashable values then returns a
    #     hash of the hashable items in the dict.
    #     """
    #     x = [(k, v) if isinstance(v, Hashable) else (k, tuple(v)) for k, v in d.items()]
    #     x = [(k, v) for k, v in x if (isinstance(k, Hashable) and isinstance(v, Hashable))]
    #     x = tuple(sorted(x))
    #     s = hex(hash(x))
    #     return s[s.index("x") + 1:]
