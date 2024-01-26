"""
Utilies to help with the output of the training process.
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.cfg import BR, OUTPUT_PATH
from src.utils import get_highest_path
from src.learn.utils import str_or_bool_to_str


@dataclass
class Args:

    model_name_or_path: str = field()
    max_length: int = field()
    task: str = field()
    depth: int = field(default=1)
    streaming: bool = field(default=True)
    exit_after_map: bool = field(default=False)
    ft_freeze_positional_embeddings: bool = field(default=False)
    ft_duplicate_positional_embeddings: bool = field(default=False)
    ft_initialize_positional_embeddings: bool = field(default=False)
    root: Path = field(default=OUTPUT_PATH)
    tail: str = field(default="")
    do_tune: bool = field(default=False)

    def __post_init__(self) -> None:
        self.ft_freeze_positional_embeddings = str_or_bool_to_str(self.ft_freeze_positional_embeddings)
        self.ft_duplicate_positional_embeddings = str_or_bool_to_str(self.ft_duplicate_positional_embeddings)
        self.ft_initialize_positional_embeddings = str_or_bool_to_str(self.ft_initialize_positional_embeddings)
        self.streaming = str_or_bool_to_str(self.streaming)
        self.exit_after_map = str_or_bool_to_str(self.exit_after_map)
        self.do_tune = str_or_bool_to_str(self.do_tune)


class OutputHelper:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int,
        task: str,
        depth: int,
        ft_freeze_positional_embeddings: bool | str,
        ft_duplicate_positional_embeddings: bool | str,
        ft_initialize_positional_embeddings: bool | str,
        root: Path,
        tail: str,
    ) -> None:
        self.root = Path(root)
        args = [
            model_name_or_path,
            str(max_length),
            task,
            str(depth),
        ]
        if task == "clf":
            args.extend(
                [
                    str(str_or_bool_to_str(ft_freeze_positional_embeddings)),
                    str(str_or_bool_to_str(ft_duplicate_positional_embeddings)),
                    str(str_or_bool_to_str(ft_initialize_positional_embeddings)),
                ]
            )
        self.path = self.root.joinpath(*args) / tail

    def __repr__(self) -> str:
        return self.path.as_posix()

    def __str__(self) -> str:
        return self.path.as_posix()

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
    def test_results_dir(self) -> Path:
        return self.path / "test_results"

    @property
    def test_results_file(self) -> Path:
        return self.test_results_dir / "results.json"

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
