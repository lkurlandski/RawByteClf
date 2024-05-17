"""
Utilies to help with the output of the training process.

FIXME:
 - add world_size to the output path instead of multiplying with per_device_train_batch_size
 - the mutability of the OutputHelper's trainer_config is confusing; 
    refactor to contain a reference to a TrainingArguments?
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import errno
import json
import os
from pathlib import Path
from pprint import pprint
import shutil
import sys
from typing import Any, Callable, Optional

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.cfg import OUTPUT_PATH
from src.utils import get_highest_path, is_jsonable
from src.learn.utils import str_or_bool_to_str, float_to_int


TASKS = [
    "clm",
    "mlm",
    "clf-bod",
    "clf-ksc",
    "clf-lxs",
    "clf-sor-org",
    "clf-sor-cat",
    "clf-sor-nam",
    "clf-sor-lab",
]
TOKENIZERS = ["Raw", "BPE", "Unigram", "WordPiece", "WordLevel", "SentencePieceBPE", "SentencePieceUnigram"]


def print_options(options: list[str]) -> str:
    return "One of " + ", ".join([f"`{x}`" for x in options[:-1]]) + f", or {options[-1]}."


def str_to_type(s: Optional[Any], t: type) -> Optional[Any]:
    if s is None:
        return None
    if isinstance(s, t) and t != str:  # Let's use the function with `t=str`
        return s
    if not isinstance(s, str):
        raise TypeError(f"Expected a string, got {type(s)}")
    if s.lower().strip() == "none":
        return None
    return t(s)


def str_to_int(s: Optional[str]) -> Optional[int]:
    return str_to_type(s, int)


def str_to_float(s: Optional[str]) -> Optional[float]:
    return str_to_type(s, float)


def str_to_str(s: Optional[str]) -> Optional[str]:
    return str_to_type(s, str)


def str_to_bool(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    if not isinstance(s, str):
        raise TypeError(f"Expected a string, got {type(s)}")

    trues = ("true", "t", "yes", "y")
    falses = ("false", "f", "no", "n")

    if s.lower() in trues:
        return True
    if s.lower() in falses:
        return False

    raise ValueError(f"Got {s}; expected one of {trues + falses}.")


@dataclass
class Args:

    model_name_or_path: str = field()
    max_length: int = field()
    task: str = field()
    representation: int = field(default=8)
    algorithm: str = field(default="Raw")
    vocab_size: Optional[int] = field(default=None)
    depth: int = field(default=1)
    streaming: bool = field(default=False)
    exit_after_map: bool = field(default=False)
    ft_freeze_positional_embeddings: bool = field(default=False)
    ft_duplicate_positional_embeddings: bool = field(default=False)
    ft_initialize_positional_embeddings: bool = field(default=False)
    root: Path = field(default=OUTPUT_PATH)
    do_tune: bool = field(default=False)
    min_freq: Optional[str] = field(default=None)
    top_k: Optional[str] = field(default=None)
    arch_config_file: Optional[Path] = field(default=None)
    arch_config: Optional[dict | str] = field(default=None)
    subset: Optional[int] = field(default=None)
    tr_size: float = field(default=0.8)
    vl_size: float = field(default=0.1)
    ts_size: float = field(default=0.1)
    skip_eval_check: bool = field(default=False)
    auto_find_batch_size_and_gradient_accumulation_steps: bool = field(default=False)
    enforce_cutoff: Optional[bool] = field(default=None)
    tr_length_cutoff: Optional[int] = field(default=None)
    early_stopping: bool = field(default=False)
    early_stopping_patience: int = field(default=1)
    early_stopping_threshold: float = field(default=0.0)
    dataset_backend: str = field(default="PT")
    data_read_bytes: Optional[int] = field(default=None)
    compression_level: int = field(default=9)
    tr_samples_per_class: Optional[int] = field(default=None)  # FIXME: make default argument 1?
    vl_samples_per_class: Optional[int] = field(default=1)  # FIXME: add to output path
    remove_packed: bool = field(default=False)
    pretraining_task: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        # Simple type conversions from string into the appropriate type.
        self.top_k = str_to_int(self.top_k)
        self.min_freq = str_to_int(self.min_freq)
        self.ft_freeze_positional_embeddings = str_to_bool(self.ft_freeze_positional_embeddings)
        self.ft_duplicate_positional_embeddings = str_to_bool(self.ft_duplicate_positional_embeddings)
        self.ft_initialize_positional_embeddings = str_to_bool(self.ft_initialize_positional_embeddings)
        self.streaming = str_to_bool(self.streaming)
        self.exit_after_map = str_to_bool(self.exit_after_map)
        self.do_tune = str_to_bool(self.do_tune)
        self.skip_eval_check = str_to_bool(self.skip_eval_check)
        self.auto_find_batch_size_and_gradient_accumulation_steps = str_to_bool(self.auto_find_batch_size_and_gradient_accumulation_steps)
        self.enforce_cutoff = str_to_bool(self.enforce_cutoff)
        self.pretraining_task = str_to_str(self.pretraining_task)

        # Parse the architecture configuration from JSON or from a file.
        if self.arch_config_file and self.arch_config:
            raise ValueError("Cannot specify both arch_config_file and arch_config.")
        if self.arch_config and isinstance(self.arch_config, str):
            self.arch_config = json.loads(self.arch_config)
            if not isinstance(self.arch_config, dict):
                raise ValueError(f"arch_config not parsed correctly: {self.arch_config=}")
        if self.arch_config_file:
            with open(self.arch_config_file) as fp:
                self.arch_config = json.load(fp)

        # Cast the train, validation, and test size to the appropriate type.
        self.tr_size = float_to_int(self.tr_size) if self.tr_size > 1 else self.tr_size
        self.vl_size = float_to_int(self.vl_size) if self.vl_size > 1 else self.vl_size
        self.ts_size = float_to_int(self.ts_size) if self.ts_size > 1 else self.ts_size
        types = [type(x) for x in [self.tr_size, self.vl_size, self.ts_size] if x > 0]
        if len(set(types)) > 1:
            raise TypeError("The semantics of using both float and int is not well defined.")
        IntOrFloat = types[0]
        self.tr_size = IntOrFloat(self.tr_size) if self.tr_size == 0.0 else self.tr_size
        self.vl_size = IntOrFloat(self.vl_size) if self.vl_size == 0.0 else self.vl_size
        self.ts_size = IntOrFloat(self.ts_size) if self.ts_size == 0.0 else self.ts_size

        # Set dependent default values.
        if self.data_read_bytes is None:
            self.data_read_bytes = int(self.max_length * self.representation // 8)


class OutputHelper:

    """
    General philosophy for organizing the various parameters is:
       - meta hyperaparameters
       - model hyperparameters
       - task hyperparameters
       - training hyperparameters
    """

    FINAL_PATH = "results"
    TRAINER_KEYS = [
        "max_grad_norm",  # Regularization
        "weight_decay",
        "learning_rate",  # Optimizer
        "lr_scheduler_type",
        "warmup_ratio",
        "optim",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_steps",  # Training
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "tf32",  # Numeric types
        "fp16",
        "bf16",
        "seed",  # Randomness
    ]

    _meta_args: list[str]
    _model_args: list[str]
    _task_args: list[str]
    _trainer_args: list[str]

    def __init__(
        self,
        root: Path,
        remove_packed: bool,
        representation: int,
        algorithm: str,
        vocab_size: Optional[int],
        max_length: int,
        model_name_or_path: str,
        arch_config: Optional[dict],
        task: str,
        tr_size: int | float,
        depth: int,
        min_freq: Optional[int],
        top_k: Optional[int],
        tr_samples_per_class: Optional[int],
        tr_length_cutoff: Optional[int],
        trainer_config: Optional[dict] = None,
        pretraining_task: Optional[str] = None,
    ) -> None:

        if pretraining_task:
            oh = OutputHelper(
                root,
                remove_packed,
                representation,
                algorithm,
                vocab_size,
                max_length,
                model_name_or_path,
                arch_config,
                pretraining_task,
                tr_size,
                depth,
                min_freq,
                top_k,
                tr_samples_per_class,
                tr_length_cutoff,
                trainer_config,
                None,
            )
            p = oh.model_path / f"task--{pretraining_task}"
            completed = list(p.rglob(OutputHelper.FINAL_PATH))
            # Ignore sub-classification experiments.
            completed = [p for p in completed if all("checkpoint-" not in part for part in p.parts)]
            if len(completed) == 0:
                raise FileNotFoundError(f"No completed experiments found for {oh.task_path=}")
            if len(completed) > 1:
                raise FileNotFoundError(f"Multiple completed experiments found for {oh.task_path=}")
            model_name_or_path = get_highest_path(completed[0] / "checkpoints", lstrip="checkpoint-").as_posix()

        if Path(model_name_or_path).exists():
            self.root = Path(model_name_or_path)
            for s in self.root.as_posix().split("/"):
                if s.startswith("model_name--"):
                    self.model_name = s[7:]
                    break
            else:
                raise ValueError(f"Could not find model_name in {self.root=}")
        else:
            self.root = root
            self.model_name = model_name_or_path

        self._meta_args = [
            f"remove_packed--{remove_packed}",
            f"representation--{representation}",
            f"algorithm--{algorithm}",
            f"vocab_size--{vocab_size if vocab_size is not None else 2 ** representation}",
            f"max_length--{max_length if max_length is not None else 'None'}",
        ]

        self._model_args = [f"model_name--{self.model_name}"]
        self._model_args.extend([f"{k}--{v}" for k, v in arch_config.items()])

        # Experiment hyperparameters
        self._task_args = [f"task--{task}"]
        if task == "clf-bod":
            self._task_args.extend([
                f"min_freq--{min_freq}",
                f"top_k--{top_k}",
            ])
        elif task == "clf-ksc":
            self._task_args.extend([
                f"tr_samples_per_class--{tr_samples_per_class}",
                f"top_k--{top_k}",
            ])
        elif task == "clf-lxs":
            self._task_args.extend([
                f"tr_length_cutoff--{tr_length_cutoff}",
            ])
        elif task[0:7] == "clf-sor":
            self._task_args.extend([
                f"min_freq--{min_freq}",
                f"top_k--{top_k}",
            ])
        elif task in ("mlm", "clm"):
            self._task_args.extend([
                f"tr_size--{tr_size}",
                f"depth--{depth}",
            ])
        else:
            raise ValueError(f"Unknown task: {task}")

        if "world_size" not in trainer_config:
            raise KeyError("world_size not found in trainer_config.")
        self._trainer_config = trainer_config
        self._trainer_args = self.get_trainer_path_args()

    def __del__(self) -> None:
        attrs = ["root", "_meta_args", "_model_args", "_task_args", "_trainer_args", "lock_file"]
        if all(hasattr(self, a) for a in attrs):
            self.lock_file.unlink(missing_ok=True)

    def __repr__(self) -> str:
        return self.path.as_posix()

    def __str__(self) -> str:
        s = ""
        for i, p in enumerate(self.path.parts):
            s += f"{' ' * (i * 2)} |-- {p}\n"

        return s

    @property
    def trainer_config(self) -> dict:
        return self._trainer_config

    @trainer_config.setter
    def trainer_config(self, config: dict) -> None:
        if "world_size" not in config:
            raise KeyError("world_size not found in trainer_config.")
        self._trainer_config = config
        self._trainer_args = self.get_trainer_path_args()

    @property
    def path(self) -> Path:
        return self.root.joinpath(
            *self._meta_args,
            *self._model_args,
            *self._task_args,
            *self._trainer_args,
        ) / OutputHelper.FINAL_PATH

    @property
    def meta_path(self) -> Path:
        return self.root.joinpath(*self._meta_args)

    @property
    def model_path(self) -> Path:
        return self.meta_path.joinpath(*self._model_args)

    @property
    def task_path(self) -> Path:
        return self.model_path.joinpath(*self._task_args)

    @property
    def trainer_path(self) -> Path:
        return self.task_path.joinpath(*self._trainer_args)

    @property
    def best_model_dir(self) -> Path:
        with open(self.last_checkpoint / "trainer_state.json") as fp:
            state = json.load(fp)
        best_model_checkpoint = state.get("best_model_checkpoint", None)
        if best_model_checkpoint is None:
            return self.last_checkpoint
        return Path(best_model_checkpoint)

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def config_file(self) -> Path:
        return self.path / "config.json"

    @property
    def test_results_dir(self) -> Path:
        return self.path / "test_results"

    @property
    def test_results_file(self) -> Path:
        return self.test_results_dir / "test_results.json"

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

    @property
    def lock_file(self) -> Path:
        return self.path / "LOCK"

    def mkdir(self) -> Path:
        # Confusingly, `exist_ok=True` and `parents=True` seems to be needed. Maybe race condition?
        ancestor: Path = None
        for p in reversed(self.path.parents):
            if not p.exists():
                p.mkdir(exist_ok=True, parents=True)
                ancestor = p if ancestor is None else ancestor

        if not self.path.exists():
            self.path.mkdir(exist_ok=True, parents=True)

        with open(self.lock_file, "w") as fp:
            fp.write("")

        ancestor = self.path if ancestor is None else ancestor
        return ancestor

    def rmdir(self, force: bool = False) -> None:
        if not self.path.exists():
            return

        if force:
            shutil.rmtree(self.path)
            return

        files = list(filter(lambda p: p.is_file(), self.path.glob("*")))
        if len(files) == 0:
            shutil.rmtree(self.path)
            return

        raise OSError(errno.ENOTEMPTY, os.strerror(errno.ENOTEMPTY), self.path)

    def get_trainer_path_args(self) -> list[str]:
        d = {k: self.trainer_config.get(k, None) for k in self.TRAINER_KEYS}
        d = {k: v.value if isinstance(v, Enum) else v for k, v in d.items()}
        d["per_device_batch_size"] = d["per_device_batch_size"] * d.pop("world_size")
        return [f"{k}--{v}" for k, v in d.items()]
