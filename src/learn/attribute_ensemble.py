"""
Code to analyze the ensemble model per major revision request.
"""

from __future__ import annotations
import gc
from pathlib import Path
import sys
from typing import Any, Generator, Iterable, Optional
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
    """
    Performs attributions for the ensemble model. Note that this class uses a batch_size of 1.
    """

    use_tqdm: bool = False

    def __init__(
        self,
        model: EnsembleForSequenceClassification,
        dataset: Dataset | IterableDataset | Iterable[dict[str, list[Tensor]]],
        data_collator: EnsembleDataCollatorWithPadding,
        baseline: int,
        device: str,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.data_collator = data_collator
        self.baseline = baseline
        self.device = device
        self.layers = [model.raw_backbone.embeddings, model.dis_backbone.embeddings, model.dec_backbone.embeddings]
        self.alg = LayerIntegratedGradients(forward_func, self.layers)
        self.skipsteps = set()

    def __iter__(self) -> Generator[tuple[int, Tensor, Tensor, Tensor, Tensor], None, None]:
        self.model.eval().to(self.device)

        iterable = enumerate(self.dataset.iter(1)) if isinstance(self.dataset, (Dataset, IterableDataset)) else enumerate(self.dataset)
        iterable = tqdm(iterable, total=len(self.dataset), desc="Explaining...") if self.use_tqdm else iterable
        print(f"Beginning explanation for {len(self.dataset)} samples.")

        t_start = time.time()
        for step, batch in iterable:
            if step in self.skipsteps:
                continue

            t_step_start      = time.time()
            cuda_oom_occurred = False

            l_raw = len(batch["raw_input_ids"][0])
            l_dis = len(batch["dis_input_ids"][0])
            l_dec = len(batch["dec_input_ids"][0])

            try:
                labels, attribs_raw, attribs_dis, attribs_dec = self.run(batch)
                yield step, labels[0], attribs_raw[0], attribs_dis[0], attribs_dec[0]
            except Exception as err:
                if not is_exception_cuda_memory_related(err):
                    raise err
                cuda_oom_occurred = True
                yield step, None, None, None, None

            del batch
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

    def run(self, batch: dict[str, list[Tensor]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch: list[dict[str, Any]]  = [dict(zip(batch, t)) for t in zip(*batch.values())]
        inputs: dict[str, list[Any]] = self.data_collator(batch)

        labels: Tensor = inputs["labels"].to(self.device)
        input_ids_raw: Tensor = inputs["raw_input_ids"][:,0:6144].to(self.device)
        input_ids_dis: Tensor = inputs["dis_input_ids"][:,0:6144].to(self.device)
        input_ids_dec: Tensor = inputs["dec_input_ids"][:,0:6144].to(self.device)

        attribs = get_attribution(self.alg, input_ids_raw, input_ids_dis, input_ids_dec, labels, self.model, self.baseline)
        attribs_raw = attribs[0].detach().to("cpu")
        attribs_dis = attribs[1].detach().to("cpu")
        attribs_dec = attribs[2].detach().to("cpu")
        labels = labels.detach().to("cpu")

        return labels, attribs_raw, attribs_dis, attribs_dec


class AttributorRunner:
    """
    Wraps the Attributor to handle input and output.

    Args:
        output (Path): The directory where the results will be saved.
        attributor (Attributor): The Attributor instance to run.
        resume (bool): If True, skips over steps that have already been completed.
        rerun (bool): If True, reruns the attributor on failed steps.
        clean (bool): If True, cleans the output directory before running.
    """

    def __init__(self, output: Path, attributor: Attributor, resume: bool = True, rerun: bool = False, clean: bool = False) -> None:
        self.output = output
        self.attributor = attributor
        self.resume = resume
        self.rerun = rerun
        self.clean = clean

    def __call__(self) -> AttributorRunner:
        if not self.output.exists():
            self.output.mkdir(parents=True, exist_ok=True)

        if self.clean:
            for f in self.output.iterdir():
                f.unlink()

        skipsteps = self.get_skipsteps()
        self.attributor.skipsteps.update(skipsteps)

        for step, labels, attribs_raw, attribs_dis, attribs_dec in iter(self.attributor):
            d = {"labels": labels, "attribs_raw": attribs_raw, "attribs_dis": attribs_dis, "attribs_dec": attribs_dec}
            f = self.output / f"step-{step:05d}.pt"
            torch.save(d, f)

    def get_skipsteps(self) -> set[int]:
        skipsteps = set()
        if self.resume:
            for f in self.output.iterdir():
                if f.is_file() and f.suffix == ".pt" and f.stem.startswith("step-"):
                    step = int(f.stem.split("-")[1])
                    if not self.rerun or torch.load(f)["attribs_raw"] is not None:
                        skipsteps.add(step)
        return skipsteps

    def print(self) -> None:
        print(f"AttributorRunner: {self.__class__.__name__}")
        print(f"\toutput: {self.output}")
        print(f"\tattributor: {self.attributor.__class__.__name__}")
        print(f"\tresume: {self.resume}")
        print(f"\trun: {self.rerun}")
        print(f"\tclean: {self.clean}")
        print(f"\tskipsteps: {self.get_skipsteps()}")
