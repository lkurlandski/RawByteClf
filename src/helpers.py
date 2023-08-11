"""
Helper classes.
"""

from __future__ import annotations
from pathlib import Path
from typing import Generator, Literal, Optional


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
                                | -- dataset
                                | -- model
                                    | -- checkpoints
                                        | -- checkpoint-{epoch}
                                            | -- config.json
                                            | ...
                                        | -- best
                                            | -- config.json
                                            | ...
                                        | -- log_history.json
    """

    def __init__(
        self,
        root: Path = "./output",
        *,
        algorithm: str = None,
        vocab_size: int = None,
        num_tok: int = None,
        max_length: int = None,
        num: float = None,
        model: str = None,
    ) -> None:
        self._root = root
        self._algorithm = algorithm
        self._vocab_size = vocab_size
        self._num_tok = num_tok
        self._max_length = max_length
        self._num = num
        self._model = model

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
    def algorithm(self) -> str:
        return str(self._algorithm)

    @property
    def vocab_size(self) -> str:
        return str(self._vocab_size) if self._vocab_size is not None else None

    @property
    def num_tok(self) -> str:
        return str(self._num_tok) if self._num_tok is not None else None

    @property
    def max_length(self) -> str:
        return str(self._max_length) if self._max_length is not None else None

    @property
    def num(self) -> str:
        return str(self._num) if self._num is not None else None

    @property
    def model(self) -> str:
        return str(self._model)

    @property
    def tokenizer_file(self) -> Path:
        parts = [self.algorithm, self.vocab_size, self.num_tok]
        if any(a is None for a in parts):
            return None
        return self.root.joinpath(*parts) / "vocab.json"

    @property
    def dataset_dir(self) -> Path:
        parts = [self.algorithm, self.vocab_size, self.num_tok, self.max_length, self.num]
        if any(a is None for a in parts):
            return None
        return self.root.joinpath(*parts) / "dataset"

    @property
    def model_dir(self) -> Path:
        parts = [
            self.algorithm,
            self.vocab_size,
            self.num_tok,
            self.max_length,
            self.num,
            self.model,
        ]
        if any(a is None for a in parts):
            return None
        return self.root.joinpath(*parts)

    @property
    def best_model_dir(self) -> Path:
        if not self.model_dir:
            return None
        return self.model_dir / "best"

    @property
    def checkpoints_dir(self) -> Path:
        if not self.model_dir:
            return None
        return self.model_dir / "checkpoints"

    @property
    def log_history_path(self) -> Path:
        if not self.model_dir:
            return None
        return self.model_dir / "log_history.json"

    @property
    def test_results_path(self) -> Path:
        if not self.model_dir:
            return None
        return self.model_dir / "test_results.json"

    def exists(self, tokenizer: bool = True, dataset: bool = True, model: bool = True) -> bool:
        return (
            (not tokenizer or (self.tokenizer_file and self.tokenizer_file.exists()))
            and (not dataset or (self.dataset_dir and self.dataset_dir.exists()))
            and (not model or (self.model_dir and self.model_dir.exists()))
        )


def iter_over_root(
    root: Path,
    *,
    algorithms: Optional[list[str]] = None,
    vocab_sizes: Optional[list[int]] = None,
    num_toks: Optional[list[int]] = None,
    max_lengths: Optional[list[int]] = None,
    nums: Optional[list[float]] = None,
    models: Optional[list[str]] = None,
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
                    for num in func(max_length, nums):
                        for model in func(num, models):
                            yield OutputHelper(
                                root=root,
                                algorithm=algorithm.name,
                                vocab_size=int(vocab_size.name),
                                num_tok=int(num_tok.name),
                                max_length=int(max_length.name),
                                num=float(num.name),
                                model=model.name,
                            )
