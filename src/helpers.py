"""
Helper classes and their associated functions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Generator, Literal, Optional


@dataclass
class OutputArgs:
    dataset: str = field()
    algorithm: str = field()
    vocab_size: int = field()
    num_tok: int = field()
    max_length: int = field()
    num_dat: int = field()
    model: str = field()
    scale: float = field()
    task: str = field()
    pretrain_task: Optional[str] = field(default=None)
    root: Path = field(default=Path("./output"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)


class BaseOutputHelper:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.path = self.root

    def mkdir(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)


class TokenizerOutputHelper(BaseOutputHelper):
    def __init__(
        self,
        root: Path,
        dataset: str,
        algorithm: str,
        vocab_size: int,
        num_tok: int,
    ) -> None:
        super().__init__(root)
        self.path = self.root.joinpath(dataset, algorithm, str(vocab_size), str(num_tok))

    @property
    def tokenizer_file(self) -> Path:
        return self.path / "tokenizer.json"


class DatasetOutputHelper(BaseOutputHelper):
    def __init__(
        self,
        root: Path,
        max_length: int,
        num_dat: Optional[float] = None,
    ) -> None:
        super().__init__(root)
        self.path = self.root.joinpath(str(max_length), str(num_dat))

    @property
    def dataset_dir(self) -> Path:
        return self.path / "dataset"


class ModelOutputHelper(BaseOutputHelper):
    def __init__(
        self,
        root: Path,
        model: str,
        scale: float,
        task: Literal["clf", "mlm", "clm"] = None,
        pretrain_task: Optional[Literal["clf", "mlm", "clm"]] = None,
    ) -> None:
        super().__init__(root)
        self.path = self.root.joinpath(model, str(scale), task, str(pretrain_task))

    @property
    def best_model_dir(self) -> Path:
        return self.path / "best_model"

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def log_history_file(self) -> Path:
        return self.path / "log_history.json"

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


class OutputHelper(BaseOutputHelper):
    def __init__(self, args: OutputArgs) -> None:
        super().__init__(args.root)
        self.tokenizer_oh = TokenizerOutputHelper(
            args.root, args.dataset, args.algorithm, args.vocab_size, args.num_tok
        )
        self.dataset_oh = DatasetOutputHelper(self.tokenizer_oh, args.max_length, args.num_dat)
        self.model_oh = ModelOutputHelper(
            self.dataset_oh, args.model, args.scale, args.task, args.pretrain_task
        )

    @property
    def path(self) -> Path:
        return self.model_oh.path

    @property
    def tokenizer_file(self) -> Path:
        return self.tokenizer_oh.tokenizer_file

    @property
    def dataset_dir(self) -> Path:
        return self.dataset_oh.dataset_dir

    @property
    def best_model_dir(self) -> Path:
        return self.model_oh.best_model_dir

    @property
    def checkpoints_dir(self) -> Path:
        return self.model_oh.checkpoints_dir

    @property
    def log_history_file(self) -> Path:
        return self.model_oh.log_history_file

    @property
    def test_results_dir(self) -> Path:
        return self.model_oh.test_results_dir

    @property
    def test_results_file(self) -> Path:
        return self.model_oh.test_labels_file

    @property
    def test_predictions_file(self) -> Path:
        return self.model_oh.test_predictions_file

    @property
    def test_probas_file(self) -> Path:
        return self.model_oh.test_probas_file

    @property
    def test_labels_file(self) -> Path:
        return self.model_oh.test_labels_file

    @property
    def test_confusion_matrix_file(self) -> Path:
        return self.model_oh.test_confusion_matrix_file

    def mkdir(self) -> None:
        super().mkdir()
        for oh in [self.tokenizer_oh, self.dataset_oh, self.model_oh]:
            oh.mkdir()


class IterOverOutputHelpers:
    def __init__(
        self,
        root: Path,
        datasets: list[str],
        algorithms: list[str],
        vocab_sizes: list[int],
        num_toks: list[int],
        max_lengths: list[int],
        num_dats: list[float],
        models: list[str],
        scales: Optional[list[float]],
        tasks: list[Literal["clf", "clm", "mlm"]],
        pretrain_tasks: list[Optional[Literal["clf", "clm", "mlm"]]],
    ) -> None:
        self.root = Path(root)
        self.datasets = datasets
        self.algorithms = algorithms
        self.vocab_sizes = vocab_sizes
        self.num_toks = num_toks
        self.max_lengths = max_lengths
        self.num_dats = num_dats
        self.models = models
        self.scales = scales
        self.tasks = tasks
        self.pretrain_tasks = pretrain_tasks

    def __iter__(self) -> Generator[OutputHelper, None, None]:
        iterable = product(
            self.func(self.root, self.datasets),
            self.func(self.datasets, self.algorithms),
            self.func(self.algorithms, self.vocab_sizes),
            self.func(self.vocab_sizes, self.num_toks),
            self.func(self.num_toks, self.max_lengths),
            self.func(self.max_lengths, self.num_dats),
            self.func(self.num_dats, self.models),
            self.func(self.models, self.scales),
            self.func(self.max_lengths, self.tasks),
            self.func(self.scales, self.pretrain_tasks),
        )
        for args in iterable:
            args = OutputArgs(*[item.name for item in args], root=self.root)
            yield OutputHelper(args)

    @staticmethod
    def iterdir(path: Path):
        for p in path.iterdir():
            if p.is_dir():
                yield p

    @staticmethod
    def func(x: Path, xs: Optional[list[Literal]]) -> list[Path]:
        return [x / str(i) for i in xs] if xs else list(IterOverOutputHelpers.iterdir(x))
