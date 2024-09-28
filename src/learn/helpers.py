"""
Utilies to help with the output of the training process.

FIXME:
 - add world_size to the output path instead of multiplying with per_device_train_batch_size
 - the mutability of the OutputHelper's trainer_config is confusing; 
    refactor to contain a reference to a TrainingArguments?

TODO:
 - after adding an update method, it seems pertinent to simply save the configuration values
  as instance variables then make the _{}_args private functions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import errno
import json
import os
from pathlib import Path
from pprint import pformat, pprint
import shutil
import sys
from typing import Any, Literal, Optional

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
else:
    print(f"Entered {__file__=}")
# pylint: enable=wrong-import-position

from src.cfg import OUTPUT_PATH
from src.utils import get_highest_path, is_jsonable
from src.data.labeling import KEYS
from src.learn.utils import str_or_bool_to_str, float_to_int


TASKS = [
    "clm-sor",
    "clm-elf",
    "mlm-sor",
    "mlm-elf",
    "clf-bod",
]
TASKS.extend([f"clf-sor-{k}" for k in KEYS])
TASKS.extend([f"clf-elf-{k}" for k in KEYS])


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


def str_to_probable_type(s: str) -> Optional[str | int | float | bool]:
    for f in (str_to_bool, str_to_int, str_to_float, str_to_str):
        try:
            return f(s)
        except ValueError:
            pass
    raise RuntimeError(f"Could not convert {s} to a type.")


@dataclass
class Args:

    # Programmatic/implementation
    root: Path = field(default=OUTPUT_PATH)
    streaming: bool = field(default=False)
    exit_after_map: bool = field(default=False)
    do_tune: bool = field(default=False)
    do_attribute: bool = field(default=False)
    skip_eval_check: bool = field(default=False)
    auto_find_batch_size_and_gradient_accumulation_steps: bool = field(default=False)
    dataset_backend: str = field(default="PT")

    # Architecture
    model_name_or_path: str = field(default="mamba")
    arch_config_file: Optional[Path] = field(default=None)
    arch_config: Optional[str] = field(default=None)

    # Data/Representation
    max_length: int = field(default=4096)
    data_read_bytes: Optional[int] = field(default=None)
    packing_protocol: str = field(default="Any")
    representation: int = field(default=8)
    lift_level: str = field(default="raw")
    algorithm: str = field(default="wdl")
    vocab_size: Optional[int] = field(default=None)
    compression_level: int = field(default=9)

    # Training/Stopping
    weighted_loss: Optional[str] = field(default=None)
    beta: Optional[float] = field(default=None)
    early_stopping: bool = field(default=False)
    early_stopping_patience: int = field(default=1)
    early_stopping_threshold: float = field(default=0.0)

    # Task-specific
    task: str = field(default="clf-bod")
    depth: int = field(default=1)
    split_mode: Optional[str] = field(default="random")
    tr_size: Optional[float] = field(default=None)
    vl_size: Optional[float] = field(default=None)
    ts_size: Optional[float] = field(default=None)
    min_freq: Optional[str] = field(default=None)  # We use str to allow for "None" (makes it easier to parse)
    top_k: Optional[str] = field(default=None)  # We use str to allow for "None" (makes it easier to parse)
    tr_samples_per_class: Optional[str] = field(default=None)  # We use str to allow for "None" (makes it easier to parse)
    max_imbalance_ratio: Optional[str] = field(default=None)  # We use str to allow for "None" (makes it easier to parse)

    # Finetuning
    pretraining_task: Optional[str] = field(default=None)
    pretraining_checkpoint: str = field(default="-1")
    ft_freeze_positional_embeddings: bool = field(default=False)
    ft_duplicate_positional_embeddings: bool = field(default=False)
    ft_initialize_positional_embeddings: bool = field(default=False)

    def __post_init__(self) -> None:
        # Simple type conversions from string into the appropriate type.
        self.top_k = str_to_int(self.top_k)
        self.min_freq = str_to_int(self.min_freq)
        self.tr_samples_per_class = str_to_int(self.tr_samples_per_class)
        self.max_imbalance_ratio = str_to_int(self.max_imbalance_ratio)

        self.ft_freeze_positional_embeddings = str_to_bool(self.ft_freeze_positional_embeddings)
        self.ft_duplicate_positional_embeddings = str_to_bool(self.ft_duplicate_positional_embeddings)
        self.ft_initialize_positional_embeddings = str_to_bool(self.ft_initialize_positional_embeddings)
        self.streaming = str_to_bool(self.streaming)
        self.exit_after_map = str_to_bool(self.exit_after_map)
        self.do_tune = str_to_bool(self.do_tune)
        self.do_attribute = str_to_bool(self.do_attribute)
        self.skip_eval_check = str_to_bool(self.skip_eval_check)
        self.auto_find_batch_size_and_gradient_accumulation_steps = str_to_bool(self.auto_find_batch_size_and_gradient_accumulation_steps)

        self.pretraining_task = str_to_str(self.pretraining_task)
        if self.pretraining_checkpoint.strip().lstrip("-").isdigit():
            self.pretraining_checkpoint = int(self.pretraining_checkpoint)
        self.weighted_loss = str_to_str(self.weighted_loss)

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
        self.tr_size = float_to_int(self.tr_size) if self.tr_size is not None and self.tr_size > 1 else self.tr_size
        self.vl_size = float_to_int(self.vl_size) if self.vl_size is not None and self.vl_size > 1 else self.vl_size
        self.ts_size = float_to_int(self.ts_size) if self.ts_size is not None and self.ts_size > 1 else self.ts_size
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

        self.packing_protocol = self.packing_protocol.lower()
        if self.packing_protocol not in ("yes", "no", "unk", "any"):
            raise ValueError(f"packing_protocol must be one of 'yes', 'no', 'unk' or 'any'. Got {self.packing_protocol=}")

        if self.split_mode not in ("random", "temporal"):
            raise ValueError(f"split_mode must be one of 'random' or 'temporal'. Got {self.split_mode=}")


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
        *,
        packing_protocol: str,
        representation: int,
        lift_level: str,
        algorithm: str,
        vocab_size: Optional[int],
        max_length: int,
        model_name_or_path: str,
        arch_config: Optional[dict],
        task: str,
        split_mode: Literal["random", "temporal"],
        tr_size: int | float,
        weighted_loss: Optional[str],
        depth: int,
        tr_samples_per_class: Optional[int],
        min_freq: Optional[int],
        top_k: Optional[int],
        max_imbalance_ratio: Optional[int],
        trainer_config: Optional[dict],
    ) -> None:

        if Path(model_name_or_path).exists():
            self.root = Path(model_name_or_path)
            for s in self.root.as_posix().split("/"):
                if s.startswith("model_name--"):
                    self.model_name = s[len("model_name--"):]
                    break
            else:
                raise ValueError(f"Could not find model_name in {self.root=}")
        else:
            self.root = root
            self.model_name = model_name_or_path

        self._meta_args = [
            f"packing_protocol--{packing_protocol}",
            f"representation--{representation}",
            f"algorithm--{algorithm}",
            f"vocab_size--{vocab_size if vocab_size is not None else 2 ** representation}",
            f"max_length--{max_length if max_length is not None else 'None'}",
        ]

        self._model_args = [f"model_name--{self.model_name}"]
        self._model_args.extend([f"{k}--{v}" for k, v in arch_config.items()])

        # Experiment hyperparameters
        self._task_args = [f"task--{task}", f"weighted_loss--{weighted_loss}", f"split_mode--{split_mode}"]
        if task[0:3] == "clf":
            self._task_args.extend([
                f"tr_size--{tr_size}",  # should be None if not doing base classification
                f"tr_samples_per_class--{tr_samples_per_class}",  # should be None if not doing few-shot
                f"top_k--{top_k}",
                f"min_freq--{min_freq}",
                f"max_imbalance_ratio--{max_imbalance_ratio}",
            ])
        elif task[0:3] in ("mlm", "clm"):
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

    def __eq__(self, other: OutputHelper) -> bool:
        return self.path == other.path

    def __repr__(self) -> str:
        return self.path.as_posix()

    def __str__(self) -> str:
        s = ""
        for i, p in enumerate(self.path.parts):
            s += f"{' ' * (i * 2)} |-- {p}\n"

        return s

    @staticmethod
    def get_finetuning_model_name_or_path(
        pretraining_task: str, pretraining_checkpoint: str | int = -1, **kwds,
    ) -> str:
        oh = OutputHelper(**kwds)

        path = oh.model_path / f"task--{pretraining_task}"
        completed = list(path.rglob(OutputHelper.FINAL_PATH))
        completed = [p for p in completed if all("checkpoint-" not in part for part in p.parts)]

        if len(completed) == 0:
            raise FileNotFoundError(f"No completed experiments found in {path=}")
        if len(completed) > 1:
            raise FileExistsError(f"Multiple completed experiments found in {path=}")

        path = completed[0] / "checkpoints"
        if isinstance(pretraining_checkpoint, int):
            model_name_or_path = get_highest_path(path, lstrip="checkpoint-", idx=pretraining_checkpoint)
        else:
            model_name_or_path = path / pretraining_checkpoint

        if not model_name_or_path.exists():
            raise FileNotFoundError(model_name_or_path.as_posix())

        return model_name_or_path.as_posix()

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
            state: dict = json.load(fp)
        if (best_model_checkpoint := state.get("best_model_checkpoint")) is None:
            return self.last_checkpoint
        return self.checkpoints_dir / Path(best_model_checkpoint).name

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
    def attribution_results_file(self) -> Path:
        return self.path / "attribution_results.pt"

    @property
    def tuning_results_dir(self) -> Path:
        return self.path / "tuning_results"

    @property
    def last_checkpoint(self) -> Path:
        return get_highest_path(self.checkpoints_dir, lstrip="checkpoint-")

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
        d["per_device_train_batch_size"] = d["per_device_train_batch_size"] * self.trainer_config["world_size"]
        return [f"{k}--{v}" for k, v in d.items()]

    def update(self, **kwds) -> None:
        collections = (
            self._meta_args,
            self._model_args,
            self._task_args,
            self._trainer_args,
        )
        for k, v in kwds.items():
            found = False
            for collection in collections:
                for i in range(len(collection)):  # pylint: disable=consider-using-enumerate
                    if f"{k}--" in collection[i]:
                        if found is False:
                            collection[i] = f"{k}--{v}"
                            found = True
                        else:
                            raise RuntimeError(f"{k=} was already found!")
            if not found:
                raise RuntimeError(f"Could not find {k=}")

    def infer_path_and_mutate(self, batch_size: bool = True, dtypes: bool = False) -> OutputHelper:
        """
        Looks for a probable path and mutates the OutputHelper to reflect the new path.
        Only able to mutate the batch size and gradient accumulation steps.
        """
        if self.path.exists():
            return self

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Could not find {self.meta_path=}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Could not find {self.model_path=}")
        if not self.task_path.exists():
            raise FileNotFoundError(f"Could not find {self.task_path=}")

        if self.trainer_path.exists():
            raise FileExistsError(f"{self.trainer_path=} already exists.")

        candidates = list(self.task_path.rglob(f"seed--{self.trainer_config['seed']}"))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No completed experiments found for {self.task_path=}")

        permitted = []
        if batch_size:
            permitted.extend(["per_device_train_batch_size", "gradient_accumulation_steps"])
        if dtypes:
            permitted.extend(["fp16", "tf32", "bf16"])

        # Examine each candidate path. If all values in the trainer_path portion are consistent,
        # then this is the correct path.
        valid = []
        for candidate in candidates:
            trainer_path_a = candidate.relative_to(self.task_path)
            trainer_path_b = self.trainer_path.relative_to(self.task_path)
            for p_a, p_b in zip(trainer_path_a.parts, trainer_path_b.parts):
                # If the keys do not match, something has gone wrong.
                k_a, v_a = p_a.split("--")
                k_b, v_b = p_b.split("--")
                if k_a != k_b:
                    raise RuntimeError(f"{k_a=} != {k_b=}")
                # If the values don't match, then the candidate is invalid.
                if v_a != v_b and k_a not in permitted:
                    break
            # If the loop didn't exit early, then the candidate is valid.
            else:
                valid.append(candidate)
        if len(valid) == 0:
            raise FileNotFoundError("No valid candidates found.")
        if len(valid) > 1:
            raise RuntimeError(f"Multiple valid candidates found:\n{pformat(valid)}\n")
        candidate = valid[0]

        trainer_path = candidate.relative_to(self.task_path)
        print(f"Found {trainer_path=}")
        for p in trainer_path.parts:
            update = False
            k, v = p.split("--")
            old = self.trainer_config[k]
            if batch_size and k in ("per_device_train_batch_size", "gradient_accumulation_steps"):
                new = int(v)
                self.trainer_config = self.trainer_config | {k: new}
                update = new != old
            elif dtypes and k in ("fp16", "tf32", "bf16"):
                new = str_to_bool(v)
                self.trainer_config = self.trainer_config | {k: new}
                update = new != old

            if update:
                print(f"{k}: {old} --> {new}")

        if not self.path.exists():
            raise FileNotFoundError(f"Could not find {self.path=}")

        return self
