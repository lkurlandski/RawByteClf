"""
MalConv CNN.

Adapted from:
https://github.com/Alexander-H-Liu/MalConv-Pytorch/tree/master
"""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from pprint import pformat, pprint
import shutil
import sys
from typing import Any, Optional

from datasets import Dataset
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerState
from tqdm import tqdm

from utils import get_highest_path


class MalConvConfig:
    def __init__(
        self,
        num_embd: int = 257,
        embed_size: int = 8,
        max_length: int = 2000000,
        window_size: int = 512,
        hidden_size: int = 128,
        num_classes: int = 2,
        pad_idx: int = 0,
        dropout_p: float = 0.5,
    ) -> None:
        self.num_embd = num_embd
        self.embed_size = embed_size
        self.max_length = max_length
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.pad_idx = pad_idx
        self.dropout_p = dropout_p

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{pformat(vars(self))}\n)"


class MalConvModel(nn.Module):
    def __init__(self, config: MalConvConfig):
        super(MalConvModel, self).__init__()
        self.embed = nn.Embedding(config.num_embd, config.embed_size, padding_idx=config.pad_idx)
        self.conv_1 = nn.Conv1d(
            int(config.embed_size / 2),
            config.hidden_size,
            config.window_size,
            stride=config.window_size,
            bias=True,
        )
        self.conv_2 = nn.Conv1d(
            int(config.embed_size / 2),
            config.hidden_size,
            config.window_size,
            stride=config.window_size,
            bias=True,
        )
        self.pooling = nn.MaxPool1d(int(config.max_length / config.window_size))
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.dropout_p),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, x: Tensor, softmax: bool = False) -> Tensor:
        x = self.embed(x)
        x = torch.transpose(x, -1, -2)
        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = F.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        x = cnn_value * gating_weight
        x = self.pooling(x)
        x = x.view(-1, 128)
        x = self.mlp(x)
        if softmax:
            x = F.softmax(x)
        return x

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(exist_ok=True)
        torch.save(self.state_dict(), save_directory / "model.pt")

    @staticmethod
    def get_state_dict(save_directory: str | Path) -> Tensor:
        save_directory = Path(save_directory)
        return torch.load(save_directory / "model.pt")


class MalConvTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        compute_metrics: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.callbacks = callbacks
        self.compute_metrics = compute_metrics
        if torch.cuda.is_available() and not self.args.no_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.loss_fn = nn.CrossEntropyLoss()
        self.state = TrainerState(epoch=0, log_history=[])

    def train(self, _) -> None:
        best_model = deepcopy(self.model.to("cpu"))
        best_accuracy = 0.0
        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        epochs = list(range(1, int(self.args.num_train_epochs) + 1))
        for epoch in tqdm(epochs, total=self.args.num_train_epochs):
            loader = self.get_dataloader(self.train_dataset, self.args.per_device_train_batch_size)
            cum_loss = 0
            self.model = self.model.train()
            self.model = self.model.to(self.device)
            for batch in enumerate(loader):
                X = batch[1]["input_ids"].to(self.device)
                Y = batch[1]["labels"].to(self.device)
                optimizer.zero_grad()
                logits = self.model(X)
                loss: Tensor = self.loss_fn(logits, Y)
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
            metrics = self.evaluate(self.eval_dataset)
            metrics["loss"] = cum_loss / len(loader)
            metrics["epoch"] = float(epoch)
            self.state.log_history.append(metrics)
            self.state.epoch = float(epoch)
            if self.args.load_best_model_at_end and metrics["eval_accuracy"] > best_accuracy:
                best_model = deepcopy(self.model.to("cpu"))
                self.model = self.model.to(self.device)
                best_accuracy = metrics["eval_accuracy"]
            path = Path(self.args.output_dir) / f"checkpoint-{epoch}"
            self.model.save_pretrained(path.as_posix())
            self.prune_checkpoints()
        if self.args.load_best_model_at_end:
            self.model = best_model.to(self.device)

    def evaluate(self, test_dataset) -> dict[str, float]:
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        loader = self.get_dataloader(test_dataset, self.args.per_device_eval_batch_size)
        losses = []
        accuracies = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                X = batch["input_ids"].to(self.device)
                Y = batch["labels"].to(self.device)
                logits = self.model(X)
                loss: Tensor = self.loss_fn(logits, Y)
                losses.append(loss.item())
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                accuracy = pred == Y
                accuracies.extend(accuracy.detach().cpu().tolist())
        return {
            "eval_loss": sum(losses) / len(losses),
            "eval_accuracy": sum(accuracies) / len(accuracies),
        }

    def get_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset.remove_columns(["text"]),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def prune_checkpoints(self) -> None:
        if self.args.save_total_limit is None:
            return
        checkpoints = list(Path(self.args.output_dir).glob("checkpoint-*"))
        if len(checkpoints) >= self.args.save_total_limit:
            checkpoint = get_highest_path(checkpoints, lstrip="checkpoint-", lowest=True)
            shutil.rmtree(checkpoint)
