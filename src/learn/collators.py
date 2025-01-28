"""
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from transformers import DataCollatorWithPadding
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase, BatchEncoding


@dataclass
class EnsembleDataCollatorWithPadding:

    # This is super fucking inefficient but I really don't care.

    raw_tokenizer: PreTrainedTokenizerBase
    dis_tokenizer: PreTrainedTokenizerBase
    dec_tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        self.raw_collator = DataCollatorWithPadding(self.raw_tokenizer, self.padding, self.max_length, self.pad_to_multiple_of, self.return_tensors)
        self.dis_collator = DataCollatorWithPadding(self.dis_tokenizer, self.padding, self.max_length, self.pad_to_multiple_of, self.return_tensors)
        self.dec_collator = DataCollatorWithPadding(self.dec_tokenizer, self.padding, self.max_length, self.pad_to_multiple_of, self.return_tensors)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        universal = [k for k in features[0].keys() if not k.startswith(("raw_", "dis_", "dec_"))]
        raw_features, dis_features, dec_features = self.split_features(features)

        raw_batch = self.raw_collator(raw_features)
        dis_batch = self.dis_collator(dis_features)
        dec_batch = self.dec_collator(dec_features)

        raw_batch = self.differentiate_batch(raw_batch, "raw_", ignore=universal)
        dis_batch = self.differentiate_batch(dis_batch, "dis_", ignore=universal)
        dec_batch = self.differentiate_batch(dec_batch, "dec_", ignore=universal)        

        data = {**raw_batch, **dis_batch, **dec_batch}
        return BatchEncoding(data)

    def split_features(self, features: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        raw = []
        dis = []
        dec = []
        for d in features:
            d_raw = {}
            d_dis = {}
            d_dec = {}
            for k, v in d.items():
                if k.startswith("raw_"):
                    d_raw[k.replace("raw_", "")] = v
                elif k.startswith("dis_"):
                    d_dis[k.replace("dis_", "")] = v
                elif k.startswith("dec_"):
                    d_dec[k.replace("dec_", "")] = v
                else:
                    d_raw[k] = v
                    d_dis[k] = v
                    d_dec[k] = v
            raw.append(d_raw)
            dis.append(d_dis)
            dec.append(d_dec)

        return raw, dis, dec

    def differentiate_batch(self, batch: dict[str, Any], s: Literal["raw_", "dis_", "dec_"], ignore: tuple[str] = tuple()) -> dict[str, Any]:
        for k in tuple(batch.keys()):
            if k not in ignore:
                batch[f"{s}{k}"] = batch.pop(k)
        return batch
