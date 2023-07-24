"""
Helper classes.
"""

from pathlib import Path


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
                                | -- {model}
                                    | -- model
                                        | -- config.json
                                        | ...
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
        self.root = Path(root)
        self.algorithm = algorithm
        self.vocab_size = str(vocab_size) if vocab_size is not None else None
        self.num_tok = str(num_tok) if num_tok is not None else None
        self.max_length = str(max_length) if max_length is not None else None
        self.num = str(num) if num is not None else None
        self.model = model

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
        return self.root.joinpath(*parts) / "model"
