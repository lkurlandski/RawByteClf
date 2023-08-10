"""
MalConv CNN.

Adapted from:
https://github.com/Alexander-H-Liu/MalConv-Pytorch/tree/master
"""

from pprint import pformat, pprint
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
from tqdm import tqdm


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
    ) -> None:
        self.num_embd = num_embd
        self.embed_size = embed_size
        self.max_length = max_length
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.pad_idx = pad_idx


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
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, x: Tensor, softmax: bool = False) -> Tensor:
        # print(f"{x.shape=}")
        x = self.embed(x)
        # print(f"{x.shape=}")
        x = torch.transpose(x, -1, -2)
        # print(f"{x.shape=}")
        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        # print(f"{cnn_value.shape=}")
        gating_weight = F.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        # print(f"{gating_weight.shape=}")
        x = cnn_value * gating_weight
        # print(f"{x.shape=}")
        x = self.pooling(x)
        # print(f"{x.shape=}")
        x = x.view(-1, 128)
        # print(f"{x.shape=}")
        x = self.mlp(x)
        # print(f"{x.shape=}")
        if softmax:
            x = F.softmax(x)
            # print(f"{x.shape=}")
        return x


class MalConvTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: DataCollatorWithPadding,
        tokenizer: PreTrainedTokenizer,
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
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and not self.args.no_cuda) else "cpu"
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, _) -> None:
        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        for epoch in tqdm(range(int(self.args.num_train_epochs)), total=self.args.num_train_epochs):
            loader = self.get_dataloader(self.train_dataset, self.args.per_device_train_batch_size)
            tr_losses = []
            for batch in enumerate(loader):
                X = batch[1]["input_ids"].to(self.device)
                Y = batch[1]["labels"].to(self.device)
                optimizer.zero_grad()
                pred = self.model(X)
                loss: Tensor = self.loss_fn(pred, Y)
                loss.backward()
                optimizer.step()
                tr_losses.append(loss.detach().cpu().numpy().tolist())
            metrics = self.evaluate(self.eval_dataset)
            metrics["vl_loss"] = metrics.pop("loss")
            metrics["vl_accuracy"] = metrics.pop("accuracy")
            metrics["tr_loss"] = sum(tr_losses) / len(tr_losses)
            metrics = {k: round(v, 3) for k, v in metrics.items()}
            print(f"{epoch}: {metrics}")

    def evaluate(self, test_dataset) -> dict[str, float]:
        loader = self.get_dataloader(test_dataset, self.args.per_device_eval_batch_size)
        losses = []
        accuracies = []
        for batch in enumerate(loader):
            X = batch[1]["input_ids"].to(self.device)
            Y = batch[1]["labels"].to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, Y)
            losses.append(loss.detach().cpu().numpy().tolist())
            pred = F.softmax(pred, dim=1).argmax(dim=1)
            accuracy = (pred == Y).sum() / len(Y)
            accuracies.append(accuracy.detach().cpu().numpy())
        return {"loss": sum(losses) / len(losses), "accuracy": sum(accuracies) / len(accuracies)}

    def get_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset.remove_columns(["text"]),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )
