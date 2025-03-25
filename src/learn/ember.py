"""
Train models on EMBER dataset.

for task in det fam beh; do
    for model in svm gbdt rf mlp; do
        python src/learn/non_neural_models.py --task $task --model $model --njobs 25
    done
done
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import os
import sys
from typing import Literal, Optional
import time

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, jaccard_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.svm import LinearSVC
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset, DataLoader

from src.enums import LiftLevel, Task
from src.data.loaders_core import Materials, ArchivedFile, get_materials_esp_det, get_materials_esp_fam, get_materials_esp_beh
from src.learn.train import CustomComputeLossFunction
from src.learn.class_weighting import sample_reweighting
from src.architectures.other import FocalLoss


@dataclass
class TabularMaterials:
    files: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    features: dict[str, np.ndarray]
    dist_tr: Optional[Counter] = None
    dist_vl: Optional[Counter] = None
    dist_ts: Optional[Counter] = None
    id2label: Optional[dict[int, int]] = None
    label2id: Optional[dict[int, int]] = None

    def __str__(self) -> str:
        files = {s: x.shape for s, x in self.files.items()}
        labels = {s: x.shape for s, x in self.labels.items()}
        features = {s: x.shape for s, x in self.features.items()}
        return f"TabularMaterials(\n\tfiles={files},\n\tlabels={labels},\n\tfeatures={features}\n)"

    def __repr__(self) -> str:
        return str(self)

    def shuffle(self) -> None:
        for s in ["tr", "vl", "ts"]:
            if len(self.files[s]) == 0:
                continue
            idx = np.random.permutation(len(self.files[s]))
            self.files[s] = self.files[s][idx]
            self.labels[s] = self.labels[s][idx]
            self.features[s] = self.features[s][idx]

    @property
    def problem_type(self) -> Optional[Literal["single_label_classification", "multi_label_classification"]]:
        if isinstance(self.labels["tr"][0], np.ndarray):
            return "multi_label_classification"
        if isinstance(self.labels["tr"], np.ndarray):
            return "single_label_classification"
        raise RuntimeError()


def multilabel_encode(labels: list[list[int]], num_classes: int) -> np.ndarray:
    y = np.zeros((len(labels), num_classes), dtype=np.int32)
    for i, label in enumerate(labels):
        for l in label:
            y[i, l] = 1
    return y


def convert_materials_for_tabular_feature_set(materials: Materials) -> TabularMaterials:
    X = np.load("./cache/ember/features.npy")
    index = Path("./cache/ember/index.txt").read_text().split("\n")
    index_to_idx = {name: idx for idx, name in enumerate(index)}

    X_tr, X_vl, X_ts, y_tr, y_vl, y_ts, s_tr, s_vl, s_ts = [], [], [], [], [], [], [], [], []

    for sha, label in zip(materials.files["tr"], materials.labels["tr"]):
        sha = sha.name.split(".")[0] if isinstance(sha, ArchivedFile) else sha.split(".")[0]
        X_tr.append(X[index_to_idx[sha]])
        y_tr.append(label)
        s_tr.append(sha)

    for sha, label in zip(materials.files["vl"], materials.labels["vl"]):
        sha = sha.name.split(".")[0] if isinstance(sha, ArchivedFile) else sha.split(".")[0]
        X_vl.append(X[index_to_idx[sha]])
        y_vl.append(label)
        s_vl.append(sha)

    for sha, label in zip(materials.files["ts"], materials.labels["ts"]):
        sha = sha.name.split(".")[0] if isinstance(sha, ArchivedFile) else sha.split(".")[0]
        X_ts.append(X[index_to_idx[sha]])
        y_ts.append(label)
        s_ts.append(sha)

    files = {
        "tr": np.array(s_tr),
        "vl": np.array(s_vl),
        "ts": np.array(s_ts),
    }

    if materials.problem_type == "multi_label_classification":
        y_tr = multilabel_encode(y_tr, materials.num_classes)
        y_vl = multilabel_encode(y_vl, materials.num_classes)
        y_ts = multilabel_encode(y_ts, materials.num_classes)
    else:
        y_tr = np.array(y_tr)
        y_vl = np.array(y_vl)
        y_ts = np.array(y_ts)

    labels = {
        "tr": y_tr,
        "vl": y_vl,
        "ts": y_ts,
    }

    features = {
        "tr": np.stack(X_tr) if len(X_tr) > 0 else np.array([]),
        "vl": np.stack(X_vl) if len(X_vl) > 0 else np.array([]),
        "ts": np.stack(X_ts) if len(X_ts) > 0 else np.array([]),
    }

    return TabularMaterials(files, labels, features, materials.dist_tr, materials.dist_vl, materials.dist_ts, materials.id2label, materials.label2id)


class MLP(nn.Module):
    """
    MLP model used in the following paper(s):
        [1] Rahman, Mohammad Saidur, et al.
            "On the limitations of continual learning for malware classification"
            Conference on Lifelong Learning Agents (CoLLAs)
            2022
        [2] Rahman, Mohammad Saidur, et al.
            "MADAR: Efficient continual learning for malware analysis with diversity-aware replay"
            arXiv preprint
            2025
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            self.block(2381, 1024),
            self.block(1024, 512),
            self.block(512, 256),
            self.block(256, 128),
            nn.Linear(128, num_classes)
        )

    def block(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


class Trainer:

    num_epochs = 5
    batch_size = 64
    warmup_ratio = 0.05
    beta = 0.990
    types = {
        Task.DET: torch.long,
        Task.FAM: torch.long,
        Task.BEH: torch.float
    }

    def __init__(self, task: Task, materials: TabularMaterials) -> None:
        self.task      = task
        self.materials = materials

        num_classes        = len(materials.dist_tr)
        num_samples        = len(materials.files["tr"])
        num_training_steps = (num_samples // self.batch_size) * self.num_epochs
        num_warmup_steps   = int(num_training_steps * self.warmup_ratio)

        self.model     = MLP(num_classes)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = LambdaLR(
            self.optimizer,
            partial(_get_linear_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        )
        self.criterion = self.create_criterion()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_criterion(self) -> nn.Module:
        if self.task == Task.DET:
            return nn.CrossEntropyLoss()
        if self.task == Task.FAM:
            weight  = sample_reweighting(self.materials.dist_tr, beta=self.beta)
            weight  = torch.tensor([weight[self.materials.id2label[i]] for i in sorted(self.materials.id2label.keys())])
            return nn.CrossEntropyLoss(weight=weight)
        if self.task == Task.BEH:
            return FocalLoss()
        raise RuntimeError()

    def fit(self, *args, **kwds) -> Trainer:  # pylint: disable=unused-argument
        self.model.to(self.device)
        self.criterion.to(self.device)

        tr_loss = self.evaluate(self.materials.features["tr"], self.materials.labels["tr"])
        vl_loss = self.evaluate(self.materials.features["vl"], self.materials.labels["vl"])
        d = self.scores(self.materials.features["vl"], self.materials.labels["vl"])
        d = {k: f"{v:.3f}" if not np.isnan(v) else v for k, v in d.items()}
        print(f"EP: {0} - LR: {self.optimizer.param_groups[0]['lr']:.5f} - TL: {tr_loss:.5f} - EL: {vl_loss:.5f} - EM: {d}")

        for epoch in range(self.num_epochs):
            self.model.train()
            dataset = TensorDataset(torch.from_numpy(self.materials.features["tr"]), torch.from_numpy(self.materials.labels["tr"]).to(self.types[self.task]))
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            l = 0.0
            n = 0
            for x, y in dataloader:
                x: Tensor = x.to(self.device)
                y: Tensor = y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss: Tensor = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                n += len(x)
                l += loss.item()

            tr_loss = l / n
            vl_loss = self.evaluate(self.materials.features["vl"], self.materials.labels["vl"])
            d = self.scores(self.materials.features["vl"], self.materials.labels["vl"])
            d = {k: f"{v:.3f}" if not np.isnan(v) else v for k, v in d.items()}
            print(f"EP: {epoch + 1} - LR: {self.optimizer.param_groups[0]['lr']:.5f} - TL: {tr_loss:.5f} - EL: {vl_loss:.5f} - EM: {d}")

        return self

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[float]:
        self.model.eval()
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).to(self.types[self.task]))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        l = 0.0
        n = 0
        for x, y in dataloader:
            x: Tensor = x.to(self.device)
            y: Tensor = y.to(self.device)
            y_pred = self.model(x)
            loss: Tensor = self.criterion(y_pred, y)
            n += len(x)
            l += loss.item()

        return l / n

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        dataset = TensorDataset(torch.from_numpy(X))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        y_pred = []
        for x, in dataloader:
            x: Tensor = x.to(self.device)
            y = self.model(x)
            if self.task in (Task.DET, Task.FAM):
                y = torch.argmax(y, dim=1).tolist()
                y_pred.extend(y)
            if self.task == Task.BEH:
                y = torch.sigmoid(y).round().int().tolist()
                y_pred.extend(y)

        return np.array(y_pred, dtype=np.int32)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != Task.DET:
            raise ValueError("Only DET task supports predict_proba")
        self.model.eval()
        dataset = TensorDataset(torch.from_numpy(X))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        y_prob = []
        for x, in dataloader:
            x: Tensor = x.to(self.device)
            y = self.model(x)
            y = torch.sigmoid(y).tolist()
            y_prob.extend(y)

        return np.array(y_prob, dtype=np.float32)

    def scores(self, X: np.ndarray, y: np.ndarray) -> dict[float]:
        if self.task == Task.DET:
            return compute_metrics_det(y, self.predict_proba(X)[:,1])
        if self.task == Task.FAM:
            return compute_metrics_fam(y, self.predict(X))
        if self.task == Task.BEH:
            return compute_metrics_beh(y, self.predict(X))
        raise RuntimeError()


def get_model_sklearn(task: Task, name: str, n_jobs: int = 1) -> HistGradientBoostingClassifier | RandomForestClassifier | LinearSVC:
    assert name in ("svm", "gbdt", "rf")

    if name == "rf":
        clf = RandomForestClassifier(n_jobs=n_jobs)
        return clf

    if name == "svm":
        clf = LinearSVC(dual="auto", max_iter=5000)
        if task == Task.DET:
            clf = CalibratedClassifierCV(clf, cv=10, n_jobs=n_jobs)
        if task == Task.FAM:
            clf = OneVsRestClassifier(clf, n_jobs=n_jobs)
        if task == Task.BEH:
            clf = MultiOutputClassifier(clf, n_jobs=n_jobs)
        return clf

    if name == "gbdt":
        clf = HistGradientBoostingClassifier()
        if task == Task.BEH:
            clf = MultiOutputClassifier(clf, n_jobs=n_jobs)
        return clf

    raise ValueError(f"Unknown model: {name}")


def compute_metrics_det(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "acc": accuracy_score(y_true, y_prob > 0.5),
        "roc": roc_auc_score(y_true, y_prob)
    }


def compute_metrics_fam(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "bac": balanced_accuracy_score(y_true, y_pred)
    }


def compute_metrics_beh(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "jac": jaccard_score(y_true, y_pred, average="macro", pos_label=1),
        "ham": hamming_loss(y_true, y_pred)
    }


def main():

    parser = ArgumentParser()
    parser.add_argument("--task", type=Task, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["svm", "gbdt", "rf", "mlp"])
    parser.add_argument("--njobs", type=int, default=1)
    args = parser.parse_args()

    TASK    = args.task
    MODEL   = args.model
    NJOBS   = args.njobs

    print("-" * 80)
    print(f"TASK:    {TASK.value}")
    print(f"MODEL:   {MODEL}")
    print(f"NJOBS:   {NJOBS}")

    lift_level = LiftLevel.NOP

    if TASK == Task.DET:
        materials = get_materials_esp_det(lift_level)
    if TASK == Task.FAM:
        materials = get_materials_esp_fam(lift_level)
    if TASK == Task.BEH:
        materials = get_materials_esp_beh(lift_level)
    print(f"Materials: {materials}")

    materials = convert_materials_for_tabular_feature_set(materials)
    materials.shuffle()
    print(f"Tabular: {materials}")

    X_tr = materials.features["tr"]
    X_vl = materials.features["vl"]
    y_tr = materials.labels["tr"]
    y_vl = materials.labels["vl"]

    if MODEL == "mlp":
        clf = Trainer(TASK, materials)
    else:
        clf = get_model_sklearn(TASK, MODEL, NJOBS)
    print(f"Classfier: {clf}")
    start = time.time()
    print("Fitting ... ")
    clf = clf.fit(X_tr, y_tr)
    print(f"Finished in {time.time() - start:.2f} seconds")

    start = time.time()
    print("Evaluating ... ")
    if TASK == Task.DET:
        prob = clf.predict_proba(X_vl)
        metrics = compute_metrics_det(y_vl, prob[:,1])
    elif TASK == Task.FAM:
        pred = clf.predict(X_vl)
        metrics = compute_metrics_fam(y_vl, pred)
    elif TASK == Task.BEH:
        pred = clf.predict(X_vl)
        metrics = compute_metrics_beh(y_vl, pred)
    print(f"Finished in {time.time() - start:.2f} seconds")

    print(metrics)
    print("-" * 80)


if __name__ == "__main__":
    main()
