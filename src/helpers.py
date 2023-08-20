"""
Helper classes and their associated functions.
"""

from __future__ import annotations
from pathlib import Path
from typing import Generator, Literal, Optional

from utils import is_dataset_path_completed


class OutputHelper:
    """
    Output helper class.

    Structure:
        |-- {root}
            |-- {algorithm}
                |-- {vocab_size}
                    |-- {num_tok}
                        |-- vocab.json
                        |-- {max_length}
                            | -- {num}
                                | -- clf
                                    | -- dataset
                                    | -- {model}
                                        | -- {scale}
                                            | -- clf
                                                | -- best
                                                    | -- config.json
                                                        ...
                                                | -- checkpoints
                                                    | -- checkpoint-{epoch}
                                                        | -- config.json
                                                            ...
                                                | -- log_history.json
                                            | -- clm
                                                ...
                                            | -- mlm
                                                ...
                                | -- clm
                                    | -- dataset
                                    | -- {model}
                                        | -- {scale}
                                            ...
                                | -- mlm
                                    | -- dataset
                                    | -- {model}
                                    | -- {scale}
                                        ...
    """

    def __init__(
        self,
        root: Path = "./output",
        *,
        algorithm: Optional[str] = None,
        vocab_size: Optional[int] = None,
        num_tok: Optional[int] = None,
        max_length: Optional[int] = None,
        task: Optional[Literal["clf", "mlm", "clm"]] = None,
        num: Optional[float] = None,
        model: Optional[str] = None,
        scale: Optional[float] = None,
        pretrain_task: Optional[Literal["clf", "mlm", "clm"]] = None,
    ) -> None:
        self._root = root
        self._algorithm = algorithm
        self._vocab_size = vocab_size
        self._num_tok = num_tok
        self._max_length = max_length
        self._task = task
        self._num = num
        self._model = model
        self._scale = scale
        self._pretrain_task = pretrain_task

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({vars(self)}) = {str(self)}"

    def __str__(self) -> str:
        if (p := self.checkpoints_dir) is not None:
            return p.as_posix()
        if (p := self.dataset_dir) is not None:
            return p.as_posix()
        if (p := self.tokenizer_file) is not None:
            return p.as_posix()
        return self.root

    @property
    def root(self) -> Path:
        return Path(self._root)

    @property
    def algorithm(self) -> Optional[str]:
        return str(self._algorithm) if self._algorithm is not None else None

    @property
    def vocab_size(self) -> Optional[str]:
        return str(self._vocab_size) if self._vocab_size is not None else None

    @property
    def num_tok(self) -> Optional[str]:
        return str(self._num_tok) if self._num_tok is not None else None

    @property
    def max_length(self) -> Optional[str]:
        return str(self._max_length) if self._max_length is not None else None

    @property
    def task(self) -> Optional[str]:
        return str(self._task) if self._task is not None else None

    @property
    def num(self) -> Optional[str]:
        return str(self._num) if self._num is not None else None

    @property
    def model(self) -> Optional[str]:
        return str(self._model) if self._model is not None else None

    @property
    def scale(self) -> Optional[str]:
        return str(self._scale) if self._scale is not None else None

    @property
    def pretrain_task(self) -> Optional[str]:
        if self._pretrain_task is not None:
            return self._pretrain_task
        if self.model is None:
            return None
        return self.model

    @property
    def tokenizer_file(self) -> Optional[Path]:
        parts = [self.algorithm, self.vocab_size, self.num_tok]
        if any(a is None for a in parts):
            return None
        return self.root.joinpath(*parts) / "vocab.json"

    @property
    def dataset_dir(self) -> Optional[Path]:
        """
        Lets all pretraining tasks share the same dataset without duplicating.
        """

        def parts(task: str) -> list:
            return [
                self.algorithm,
                self.vocab_size,
                self.num_tok,
                self.max_length,
                task,
                self.num,
            ]

        if any(a is None for a in parts(self.task)):
            return None
        if self.task == "clf":
            return self.root.joinpath(*(parts(self.task))) / "dataset"

        for task in ("mlm", "clm"):
            path = self.root.joinpath(*(parts(task))) / "dataset"
            if is_dataset_path_completed(path):
                return path

        return self.root.joinpath(*parts(self.task)) / "dataset"

    @property
    def clf_model_dir(self) -> Optional[Path]:
        parts = [
            self.algorithm,
            self.vocab_size,
            self.num_tok,
            self.max_length,
            self.task,
            self.num,
            self.model,
        ]
        if any(a is None for a in parts):
            return None
        return self.root.joinpath(*parts)

    @property
    def scale_model_dir(self) -> Optional[Path]:
        if not (self.clf_model_dir and self.scale):
            return None
        return self.clf_model_dir / self.scale

    @property
    def model_dir(self) -> Optional[Path]:
        if not (self.scale_model_dir and self.pretrain_task):
            return None
        return self.scale_model_dir / self.pretrain_task

    @property
    def best_model_dir(self) -> Optional[Path]:
        if not self.model_dir:
            return None
        return self.model_dir / "best"

    @property
    def checkpoints_dir(self) -> Optional[Path]:
        if not self.model_dir:
            return None
        return self.model_dir / "checkpoints"

    @property
    def log_history_path(self) -> Optional[Path]:
        if not self.model_dir:
            return None
        return self.model_dir / "log_history.json"

    @property
    def test_results_path(self) -> Optional[Path]:
        if not self.model_dir:
            return None
        return self.model_dir / "test_results.json"

    def exists(self, tokenizer: bool = True, dataset: bool = True, model: bool = True) -> bool:
        return (
            (not tokenizer or (self.tokenizer_file and self.tokenizer_file.exists()))
            and (not dataset or (self.dataset_dir and self.dataset_dir.exists()))
            and (not model or (self.model_dir and self.model_dir.exists()))
        )

    def mkdir(self, exist_ok: bool = False) -> None:
        dirs = [
            self.tokenizer_file.parent,
            self.dataset_dir,
            self.clf_model_dir,
            self.scale_model_dir,
            self.model_dir,
            self.best_model_dir,
            self.checkpoints_dir,
        ]
        for p in dirs:
            if p is not None:
                p.mkdir(exist_ok=exist_ok, parents=True)


def iter_over_root(
    root: Path,
    *,
    algorithms: Optional[list[str]] = None,
    vocab_sizes: Optional[list[int]] = None,
    num_toks: Optional[list[int]] = None,
    max_lengths: Optional[list[int]] = None,
    tasks: Optional[list[Literal["clf", "clm", "mlm"]]] = None,
    nums: Optional[list[float]] = None,
    models: Optional[list[str]] = None,
    scales: Optional[list[float]] = None,
    pretrain_tasks: Optional[list[Literal["clf", "clm", "mlm"]]],
) -> Generator[OutputHelper, None, None]:
    def iterdir(path: Path):
        for p in path.iterdir():
            if p.is_dir():
                yield p

    def func(x: Path, xs: Optional[list[Literal]]) -> list[Path]:
        return [x / str(i) for i in xs] if xs else list(iterdir(x))

    for algorithm in func(root, algorithms):
        for vocab_size in func(algorithm, vocab_sizes):
            for num_tok in func(vocab_size, num_toks):
                for max_length in func(num_tok, max_lengths):
                    for task in func(max_length, tasks):
                        for num in func(max_length, nums):
                            for model in func(num, models):
                                for scale in func(model, scales):
                                    for pretrain_task in func(scale, pretrain_tasks):
                                        yield OutputHelper(
                                            root=root,
                                            algorithm=algorithm.name,
                                            vocab_size=int(vocab_size.name),
                                            num_tok=int(num_tok.name),
                                            max_length=int(max_length.name),
                                            task=task.name,
                                            num=float(num.name),
                                            model=model.name,
                                            scale=float(scale.name),
                                            pretrain_task=pretrain_task.name,
                                        )
