"""
Code to analyze the ensemble model per major revision request.
"""

from __future__ import annotations
import gc
from pathlib import Path
import sys
from typing import Any, Optional
import time
import warnings

from captum.attr import LayerIntegratedGradients
from datasets import Dataset, IterableDataset
from torch import Tensor
import torch
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm import tqdm

from src.architectures.ensemble import EnsembleForSequenceClassification
from src.learn.collators import EnsembleDataCollatorWithPadding
from src.learn.helpers import OutputHelper
from src.learn.utils import clear_cuda_caches, should_reduce_batch_size as is_exception_cuda_memory_related
from src.utils import get_highest_path


def get_model_name_or_path(oh: OutputHelper) -> str:
    oh._task_args.pop(-1)
    if oh.checkpoints_dir.exists():
        return str(oh.last_checkpoint)
    try:
        print("A checkpoint was not found in output helper path. Mutating the dtypes in an attempt to resolve this.")
        oh = oh.infer_path_and_mutate(batch_size=False, dtypes=True)
    except FileNotFoundError:
        print("A checkpoint was not found in output helper path. Mutating the batch_size in an attempt to resolve this.")
        oh = oh.infer_path_and_mutate(batch_size=True, dtypes=True)
    return str(oh.last_checkpoint)


def forward_func(
    raw_input_ids: Tensor,
    dis_input_ids: Tensor,
    dec_input_ids: Tensor,
    model: EnsembleForSequenceClassification,
    targets: Optional[Tensor] = None,
) -> Tensor:
    output: SequenceClassifierOutput = model.forward(raw_input_ids=raw_input_ids, dis_input_ids=dis_input_ids, dec_input_ids=dec_input_ids)
    logits = output.logits
    probas = F.softmax(logits, dim=1)
    if targets is not None:
        return (probas * targets).sum(dim=1)
    return probas


def get_attribution(
    alg: LayerIntegratedGradients,
    raw_input_ids: Tensor,
    dis_input_ids: Tensor,
    dec_input_ids: Tensor,
    labels: Tensor,
    model: PreTrainedModel,
    baseline: int,
) -> tuple[Tensor, Tensor, Tensor]:
    inputs = (raw_input_ids, dis_input_ids, dec_input_ids)
    baselines_raw = torch.full_like(raw_input_ids, baseline, dtype=torch.long)
    baselines_dis = torch.full_like(dis_input_ids, baseline, dtype=torch.long)
    baselines_dec = torch.full_like(dec_input_ids, baseline, dtype=torch.long)
    baselines = (baselines_raw, baselines_dis, baselines_dec)
    attribs = alg.attribute(inputs, baselines, labels, (model,), internal_batch_size=raw_input_ids.shape[0])
    attribs_raw = attribs[0].to(torch.float32).mean(dim=-1)
    attribs_dis = attribs[1].to(torch.float32).mean(dim=-1)
    attribs_dec = attribs[2].to(torch.float32).mean(dim=-1)
    return attribs_raw, attribs_dis, attribs_dec


class Attributor:

    use_tqdm: bool = False

    def __init__(
        self,
        model: EnsembleForSequenceClassification,
        dataset: Dataset,
        data_collator: EnsembleDataCollatorWithPadding,
        baseline: int,
        device: str,
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.data_collator = data_collator
        self.baseline = baseline
        self.device = device
        self.batch_size = batch_size
        self.layers = [model.raw_backbone.embeddings, model.dis_backbone.embeddings, model.dec_backbone.embeddings]
        self.alg = LayerIntegratedGradients(forward_func, self.layers)

    def __call__(self) -> Attributor:
        self.model.eval().to(self.device)

        all_attribs_raw: list[Optional[Tensor]] = []
        all_attribs_dis: list[Optional[Tensor]] = []
        all_attribs_dec: list[Optional[Tensor]] = []
        all_labels:      list[Optional[Tensor]] = []

        total = len(self.dataset) // self.batch_size + 1
        iterable = enumerate(self.dataset.iter(self.batch_size))
        iterable = tqdm(iterable, total=total, desc="Explaining...") if self.use_tqdm else iterable
        print(f"Beginning explanation for {len(self.dataset)} samples in {total} batches.")

        t_start = time.time()
        for step, batch in iterable:

            t_step_start      = time.time()
            cuda_oom_occurred = False
            batch_size        = len(batch["labels"])

            l_raw = len(batch["raw_input_ids"][0])
            l_dis = len(batch["dis_input_ids"][0])
            l_dec = len(batch["dec_input_ids"][0])

            try:
                labels, attribs_raw, attribs_dis, attribs_dec = self.run(batch)
            except Exception as err:
                if not is_exception_cuda_memory_related(err):
                    raise err

                cuda_oom_occurred = True

                labels      = [None] * batch_size
                attribs_raw = [None] * batch_size
                attribs_dis = [None] * batch_size
                attribs_dec = [None] * batch_size

            all_labels.extend([l for l in labels])            # pylint: disable=unnecessary-comprehension
            all_attribs_raw.extend([a for a in attribs_raw])  # pylint: disable=unnecessary-comprehension
            all_attribs_dis.extend([a for a in attribs_dis])  # pylint: disable=unnecessary-comprehension
            all_attribs_dec.extend([a for a in attribs_dec])  # pylint: disable=unnecessary-comprehension

            del batch
            del labels
            del attribs_raw
            del attribs_dis
            del attribs_dec

            if cuda_oom_occurred:
                self.model.to("cpu")
            gc.collect()
            clear_cuda_caches(verbose=False)
            self.model.to(self.device)       

            d = {
                "step": step,
                "success": not cuda_oom_occurred,
                "len_raw": l_raw,
                "len_dis": l_dis,
                "len_dec": l_dec,
                "t_step": round(time.time() - t_step_start, 2),
                "t_total": round(time.time() - t_start, 2),
            }
            print(d, flush=True)

        print(f"Total time: {round(time.time() - t_start, 2)} seconds", flush=True)
        return self

    def run(self, batch: dict[str, list[Tensor]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch: list[dict[str, Any]]  = [dict(zip(batch, t)) for t in zip(*batch.values())]
        inputs: dict[str, list[Any]] = self.data_collator(batch)

        labels: Tensor = inputs["labels"].to(self.device)
        input_ids_raw: Tensor = inputs["raw_input_ids"].to(self.device)
        input_ids_dis: Tensor = inputs["dis_input_ids"].to(self.device)
        input_ids_dec: Tensor = inputs["dec_input_ids"].to(self.device)

        attribs = get_attribution(self.alg, input_ids_raw, input_ids_dis, input_ids_dec, labels, self.model, self.baseline)
        attribs_raw = attribs[0].detach().to("cpu")
        attribs_dis = attribs[1].detach().to("cpu")
        attribs_dec = attribs[2].detach().to("cpu")
        labels = labels.detach().to("cpu")

        return labels, attribs_raw, attribs_dis, attribs_dec


def _get_resume_step(self) -> int:
    if not self.output.exists():
        return 0
    try:
        highest_path = get_highest_path(self.output, "attribs", ".pt")
    except FileNotFoundError:
        return 0
    return int(highest_path.stem.split("_")[-1]) + 1
