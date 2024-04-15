"""
A huggingface-compatible implementation of MalConv2 and MalConvGCG.

# TODO: implement a way to load the models from disk.
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional, Literal
import warnings

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# pylint: enable=wrong-import-position

import numpy as np
import safetensors
import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_NAME

from src.utils import object_from_superset_of_constructor_kwds

# The default configuration values are determined by the value in the training scripts
# or from the default values of the __init__ function from the original implementation.
# These defaults may not work that well, but they are what seems to have been used
# in the original codebase and the original paper. There are some suggested hyperparameter
# values from tuning experiments in some of the dictionaries below.
# For compatibility with PretrainedConfig, all values must have a default, hence the -1s.
# For classification with MalConv or MalConvGCG, set out_size to the number of classes.


class MyMalConvConfig(PretrainedConfig):
    def __init__(
        self,
        out_size: int = -1,
        pad_idx: int = -1,
        num_embd: int = -1,
        max_length: int = -1,
        embd_size: int = 8,
        hidden_size: int = -1,
        window_size: int = 512,
        channels: int = 128,
        stride: int = 512,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        num_labels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.pad_idx = pad_idx
        self.num_embd = num_embd
        self.max_length = max_length
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.channels = channels
        self.stride = stride
        self.id2label = id2label
        self.label2id = label2id

        if num_labels is not None and out_size != num_labels:
            raise ValueError(
                f"{num_labels=} is an alias for {out_size=}. If both are passed as an argument, "
                "they must be equal. Else, just pass `out_size` and `num_labels` will be set to it."
            )
        self.num_labels = out_size


class BaseMalConvConfig(PretrainedConfig):
    def __init__(
        self,
        out_size: int = -1,
        pad_idx: int = -1,
        num_embd: int = -1,
        embd_size: int = -1,
        hidden_size: int = -1,
        window_size: int = -1,
        channels: int = -1,
        stride: int = -1,
        chunk_size: int = 65536,
        overlap: int = 512,
        min_chunk_size: int = 1024,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        num_labels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.pad_idx = pad_idx
        self.num_embd = num_embd
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.channels = channels
        self.stride = stride
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.id2label = id2label
        self.label2id = label2id

        if num_labels is not None and out_size != num_labels:
            raise ValueError(
                f"{num_labels=} is an alias for {out_size=}. If both are passed as an argument, "
                "they must be equal. Else, just pass `out_size` and `num_labels` will be set to it."
            )
        self.num_labels = out_size

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            warnings.warn(
                "MalConv does not support multi-GPU training. "
                "Use CUDA_VISIIBLE_DEVICES=0 python ... when running the script. "
                "Alternatively, use --no_cuda or --use_cpu to run on CPU."
            )


class MalConvConfig(BaseMalConvConfig):
    def __init__(
        self,
        out_size: int = -1,
        pad_idx: int = -1,
        num_embd: int = -1,
        embd_size: int = 8,
        hidden_size: int = -1,
        window_size: int = 512,
        channels: int = 128,
        stride: int = 512,
        chunk_size: int = 65536,
        overlap: int = 512,
        min_chunk_size: int = 1024,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        num_labels: Optional[int] = None,
    ) -> None:
        super().__init__(
            out_size=out_size,
            pad_idx=pad_idx,
            num_embd=num_embd,
            embd_size=embd_size,
            hidden_size=hidden_size,
            window_size=window_size,
            channels=channels,
            stride=stride,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            id2label=id2label,
            label2id=label2id,
            num_labels=num_labels,
        )


class MalConvMLConfig(BaseMalConvConfig):
    def __init__(
        self,
        out_size: int = -1,
        pad_idx: int = -1,
        num_embd: int = -1,
        embd_size: int = 8,
        hidden_size: int = -1,
        window_size: int = 512,
        channels: int = 128,
        stride: int = 512,
        layers: int = 1,
        chunk_size: int = 65536,
        overlap: int = 512,
        min_chunk_size: int = 1024,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        num_labels: Optional[int] = None,
    ) -> None:
        super().__init__(
            out_size=out_size,
            pad_idx=pad_idx,
            num_embd=num_embd,
            embd_size=embd_size,
            hidden_size=hidden_size,
            window_size=window_size,
            channels=channels,
            stride=stride,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            id2label=id2label,
            label2id=label2id,
            num_labels=num_labels,
        )
        self.layers = layers


class MalConvGCTConfig(BaseMalConvConfig):
    def __init__(
        self,
        out_size: int = -1,
        pad_idx: int = -1,
        num_embd: int = -1,
        embd_size: int = 8,
        hidden_size: int = -1,
        window_size: int = 64,
        channels: int = 128,
        stride: int = 64,
        layers: int = 1,
        chunk_size: int = 65536,
        overlap: int = 512,
        min_chunk_size: int = 1024,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        num_labels: Optional[int] = None,
    ) -> None:
        super().__init__(
            out_size=out_size,
            pad_idx=pad_idx,
            num_embd=num_embd,
            embd_size=embd_size,
            hidden_size=hidden_size,
            window_size=window_size,
            channels=channels,
            stride=stride,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            id2label=id2label,
            label2id=label2id,
            num_labels=num_labels,
        )
        self.layers = layers


def config_from_json(path: Path) -> BaseMalConvConfig | MyMalConvConfig:
    with open(path, "r") as fp:
        config: dict = json.load(fp)

    if (config_class := config.pop("config_class", None)) is not None:
        if config_class == "MalConvConfig":
            return object_from_superset_of_constructor_kwds(MalConvConfig, **config)
        if config_class == "MalConvMLConfig":
            return object_from_superset_of_constructor_kwds(MalConvMLConfig, **config)
        if config_class == "MalConvGCTConfig":
            return object_from_superset_of_constructor_kwds(MalConvGCTConfig, **config)
        if config_class == "MyMalConvConfig":
            return object_from_superset_of_constructor_kwds(MyMalConvConfig, **config)
        raise RuntimeError(f"Unknown config class: {config_class}")

    if (model_class := config.pop("architectures", None)) is not None:
        if model_class == "MalConv":
            return object_from_superset_of_constructor_kwds(MalConvConfig, **config)
        if model_class == "MalConvML":
            return object_from_superset_of_constructor_kwds(MalConvMLConfig, **config)
        if model_class == "MalConvGCT":
            return object_from_superset_of_constructor_kwds(MalConvGCTConfig, **config)
        if model_class == "MyMalConv":
            return object_from_superset_of_constructor_kwds(MyMalConvConfig, **config)
        raise RuntimeError(f"Unknown model class: {model_class}")

    raise RuntimeError("Could not process config file. Neither `config_class` or `architectures` keywords found.")


def model_from_config(config: BaseMalConvConfig | MyMalConvConfig) -> nn.Module:
    if isinstance(config, MalConvConfig):
        return MalConv(config)
    if isinstance(config, MalConvMLConfig):
        return MalConvML(config)
    if isinstance(config, MalConvGCTConfig):
        return MalConvGCT(config)
    if isinstance(config, MyMalConvConfig):
        return MyMalConv(config)

    raise RuntimeError(f"Unknown config class: {type(config)}")


class AutoMalConvForSequenceClassification:
    @staticmethod
    def from_config(config: BaseMalConvConfig | MyMalConvConfig) -> nn.Module:
        return model_from_config(config)

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, *args, **kwds) -> nn.Module:
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config = config_from_json(pretrained_model_name_or_path / CONFIG_NAME)
        model = model_from_config(config)
        if (state_dict_file := pretrained_model_name_or_path / SAFE_WEIGHTS_NAME).exists():
            model_state_dict = safetensors.torch.load_file(state_dict_file)
        elif (state_dict_file := pretrained_model_name_or_path / WEIGHTS_NAME).exists():
            model_state_dict = torch.load(state_dict_file)
        else:
            raise FileNotFoundError(
                f"Could not find {SAFE_WEIGHTS_NAME} or {WEIGHTS_NAME} " f"in {pretrained_model_name_or_path}"
            )

        model.load_state_dict(model_state_dict, strict=False)
        return model


# pylint: disable=unused-argument
def drop_zeros_hook(module: Any, grad_input: list[Tensor], grad_out: Any) -> tuple[Tensor]:
    """
    This function is used to replace gradients that are all zeros with None
    In pyTorch None will not get back-propogated
    So we use this as a approximation to saprse BP to avoid redundant and useless work
    """
    grads = []
    with torch.no_grad():
        for g in grad_input:
            if torch.nonzero(g).shape[0] == 0:
                grads.append(g.to_sparse())
            else:
                grads.append(g)

    return tuple(grads)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            ctx.input_tensors[i] = temp.detach()
            ctx.input_tensors[i].requires_grad = temp.requires_grad
        with torch.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        input_grads = torch.autograd.grad(
            output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True
        )
        return (None, None) + input_grads


class CatMod(torch.nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=2)


class LowMemConvBase(nn.Module, ABC):
    def __init__(self, chunk_size: int, overlap: int, min_chunk_size: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        # Used for pooling over time in a meory efficent way
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.cat = CatMod()
        self.cat.register_backward_hook(drop_zeros_hook)
        self.receptive_field = None

        # Used to force checkpoint code to behave correctly due to poor design
        # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    @abstractmethod
    def processRange(self, x: Tensor, **kwargs) -> None:
        """
        This method does the work to convert an LongTensor input x of shape (B, L) , where B is the
        batch size and L is the length of the input. The output of this functoin should be a tensor
        of (B, C, L), where C is the number of channels, and L is again the input length
        (though its OK if it got a little shorter due to convs without padding or something).
        """

    def determinRF(self) -> tuple[int]:
        """
        Lets determine the receptive field & stride of our sub-network
        """

        if self.receptive_field is not None:
            return self.receptive_field, self.stride, self.out_channels
        # else, figure this out!

        if not hasattr(self, "device_ids"):
            # We are training with just one device. Lets find out where we should move the data
            cur_device = next(self.embd.parameters()).device
        else:
            cur_device = "cpu"

        # Lets do a simple binary search to figure out how large our RF is.
        # It can't be larger than our chunk size! So use that as upper bound
        min_rf = 1
        max_rf = self.chunk_size

        with torch.no_grad():
            tmp = torch.zeros((1, max_rf)).long().to(cur_device)

            while True:
                test_size = (min_rf + max_rf) // 2
                is_valid = True
                try:
                    self.processRange(tmp[:, 0:test_size])
                except:
                    is_valid = False

                if is_valid:
                    max_rf = test_size
                else:
                    min_rf = test_size + 1

                if max_rf == min_rf:
                    self.receptive_field = min_rf
                    out_shape = self.processRange(tmp).shape
                    self.stride = self.chunk_size // out_shape[2]
                    self.out_channels = out_shape[1]
                    break

        return self.receptive_field, self.stride, self.out_channels

    def pool_group(self, *args) -> Tensor:
        x = self.cat(args)
        x = self.pooling(x)
        return x

    def seq2fix(self, x: Tensor, pr_args: Optional[dict] = None) -> Tensor:
        """
        Takes in an input LongTensor of (B, L) that will be converted to a fixed length
        representation (B, C), where C is the number of channels provided by the base_network
        given at construction.
        """
        pr_args = {} if pr_args is None else pr_args

        receptive_window, stride, out_channels = self.determinRF()

        if x.shape[1] < receptive_window:  # This is a tiny input! pad it out please
            x = F.pad(x, (0, receptive_window - x.shape[1]), value=self.config.pad_idx)

        batch_size = x.shape[0]
        length = x.shape[1]

        # Lets go through the input data without gradients first, and find the positions that "win"
        # the max-pooling. Most of the gradients will be zero, and we don't want to waste valuable
        # memory and time computing them.
        # Once we know the winners, we will go back and compute the forward activations on JUST
        # the subset of positions that won!
        winner_values = np.zeros((batch_size, out_channels)) - 1.0
        winner_indices = np.zeros((batch_size, out_channels), dtype=np.int64)

        if not hasattr(self, "device_ids"):
            cur_device = next(self.embd.parameters()).device
        else:
            cur_device = None

        step = self.chunk_size
        start = 0
        end = start + step

        with torch.no_grad():
            while start < end and (end - start) >= max(self.min_chunk_size, receptive_window):
                x_sub = x[:, start:end]
                if cur_device is not None:
                    x_sub = x_sub.to(cur_device)
                activs = self.processRange(x_sub.long(), **pr_args)
                activ_win, activ_indx = F.max_pool1d(activs, kernel_size=activs.shape[2], return_indices=True)
                activ_win = activ_win.cpu().numpy()[:, :, 0]
                activ_indx = activ_indx.cpu().numpy()[:, :, 0]
                selected = winner_values < activ_win
                winner_indices[selected] = activ_indx[selected] * stride + start
                winner_values[selected] = activ_win[selected]
                start = end
                end = min(start + step, length)

        # Now we know every index that won, we need to compute values and with gradients!

        # Find unique winners for every batch
        final_indices = [np.unique(winner_indices[b, :]) for b in range(batch_size)]

        # Collect inputs that won for each batch
        chunk_list = [
            [x[b : b + 1, max(i - receptive_window, 0) : min(i + receptive_window, length)] for i in final_indices[b]]
            for b in range(batch_size)
        ]
        # Convert to a torch tensor of the bytes
        chunk_list = [torch.cat(c, dim=1)[0, :] for c in chunk_list]

        # Padd out shorter sequences to the longest one
        x_selected = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)

        # Shape is not (B, L) Lets compute!

        if cur_device is not None:
            x_selected = x_selected.to(cur_device)
        x_selected = self.processRange(x_selected.long(), **pr_args)
        x_selected = self.pooling(x_selected)
        x_selected = x_selected.view(x_selected.size(0), -1)

        return x_selected


class MalConv(LowMemConvBase):
    def __init__(self, config: MalConvConfig) -> None:
        super().__init__(config.chunk_size, config.overlap, config.min_chunk_size)

        self.config = config

        self.embd = nn.Embedding(config.num_embd, config.embd_size, padding_idx=self.config.pad_idx)
        self.conv_1 = nn.Conv1d(config.embd_size, config.channels, config.window_size, stride=config.stride, bias=True)
        self.conv_2 = nn.Conv1d(config.embd_size, config.channels, config.window_size, stride=config.stride, bias=True)

        self.mlp = ClassificationHead(
            config.channels,
            config.num_labels,
            config.hidden_size,
            "leaky_relu",
            0.5,
        )

    def processRange(self, x: Tensor) -> Tensor:
        x = self.embd(x)
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))

        x = cnn_value * gating_weight

        return x

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> SequenceClassifierOutput:
        x = input_ids

        post_conv = self.seq2fix(x)
        logits = self.mlp(post_conv)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class MalConvML(LowMemConvBase):
    def __init__(self, config: MalConvMLConfig) -> None:
        super().__init__(config.chunk_size, config.overlap, config.min_chunk_size)
        self.config = config

        self.embd = nn.Embedding(config.num_embd, config.embd_size, padding_idx=self.config.pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    config.embd_size,
                    config.channels * 2,
                    config.window_size,
                    stride=config.stride,
                    bias=True,
                )
            ]
            + [
                nn.Conv1d(config.channels, config.channels * 2, config.window_size, stride=1, bias=True)
                for _ in range(config.layers - 1)
            ]
        )
        self.convs_1 = nn.ModuleList(
            [nn.Conv1d(config.channels, config.channels, 1, bias=True) for i in range(config.layers)]
        )

        self.fc_1 = nn.Linear(config.channels, config.channels)
        self.fc_2 = nn.Linear(config.channels, config.out_size)

    def processRange(self, x: Tensor) -> Tensor:
        x = self.embd(x)
        x = x.permute(0, 2, 1).contiguous()

        for conv_glu, conv_share in zip(self.convs, self.convs_1):
            x = F.leaky_relu(conv_share(F.glu(conv_glu(x.contiguous()), dim=1)))

        return x

    def forward(self, x: Tensor) -> tuple[Tensor]:
        post_conv = x = self.seq2fix(x)
        penult = x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x, penult, post_conv


class MalConvGCT(LowMemConvBase):
    def __init__(self, config: MalConvGCTConfig) -> None:
        super().__init__(config.chunk_size, config.overlap, config.min_chunk_size)
        self.config = config

        self.low_mem = True
        self.embd = nn.Embedding(config.num_embd, config.embd_size, padding_idx=self.config.pad_idx)

        self.context_net = MalConvML(
            MalConvMLConfig(
                num_embd=config.num_embd,
                hidden_size=config.hidden_size,
                out_size=config.channels,
                channels=config.channels,
                window_size=config.window_size,
                stride=config.stride,
                layers=config.layers,
                embd_size=config.embd_size,
                pad_idx=config.pad_idx,
                chunk_size=config.chunk_size,
                overlap=config.overlap,
                min_chunk_size=config.min_chunk_size,
            )
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    config.embd_size,
                    config.channels * 2,
                    config.window_size,
                    stride=config.stride,
                    bias=True,
                )
            ]
            + [
                nn.Conv1d(config.channels, config.channels * 2, config.window_size, stride=1, bias=True)
                for _ in range(config.layers - 1)
            ]
        )

        self.linear_atn = nn.ModuleList([nn.Linear(config.channels, config.channels) for _ in range(config.layers)])

        # one-by-one cons to perform information sharing
        self.convs_share = nn.ModuleList(
            [nn.Conv1d(config.channels, config.channels, 1, bias=True) for _ in range(config.layers)]
        )

        self.mlp = ClassificationHead(
            config.channels,
            config.num_labels,
            config.hidden_size,
            "leaky_relu",
            0.5,
        )

    def determinRF(self) -> tuple[int]:
        """Over-write the determinRF call to use the base context_net to detemrin RF.
        We should have the same total RF, and this will simplify logic significantly."""
        return self.context_net.determinRF()

    def processRange(self, x: int, gct: Tensor) -> Tensor:
        if gct is None:
            raise Exception("No Global Context Given")

        x = self.embd(x)
        # x = torch.transpose(x,-1,-2)
        x = x.permute(0, 2, 1)

        for conv_glu, linear_cntx, conv_share in zip(self.convs, self.linear_atn, self.convs_share):
            x = F.glu(conv_glu(x), dim=1)
            x = F.leaky_relu(conv_share(x))
            x_len = x.shape[2]
            B = x.shape[0]
            C = x.shape[1]

            sqrt_dim = np.sqrt(x.shape[1])
            # we are going to need a version of GCT with a time dimension,
            # which we will adapt as needed to the right length
            ctnx = torch.tanh(linear_cntx(gct))

            # Size is (B, C), but we need (B, C, 1) to use as a 1d conv filter
            ctnx = torch.unsqueeze(ctnx, dim=2)
            # roll the batches into the channels
            x_tmp = x.view(1, B * C, -1)
            # Now we can apply a conv with B groups, so that each batch gets
            # its own context applied only to what was needed
            x_tmp = F.conv1d(x_tmp, ctnx, groups=B)
            # x_tmp will have a shape of (1, B, L), now we just need to
            # re-order the data back to (B, 1, L)
            x_gates = x_tmp.view(B, 1, -1)

            # Now we effectively apply σ(x_t^T tanh(W c))
            gates = torch.sigmoid(x_gates)
            x = x * gates

        return x

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> SequenceClassifierOutput:
        x = input_ids

        if self.low_mem:
            global_context = CheckpointFunction.apply(self.context_net.seq2fix, 1, x)
        else:
            global_context = self.context_net.seq2fix(x)

        post_conv = self.seq2fix(x, pr_args={"gct": global_context})
        logits = self.mlp(post_conv)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class MyMalConv(nn.Module):
    """
    Adapted from https://github.com/Alexander-H-Liu/MalConv-Pytorch/tree/master
    and https://github.com/elastic/ember/blob/master/malconv/malconv.py
    """

    def __init__(self, config: MyMalConvConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.num_embd, config.embd_size, padding_idx=config.pad_idx)
        self.conv_1 = nn.Conv1d(
            int(config.embd_size / 2),
            config.channels,
            config.window_size,
            stride=config.window_size,
            bias=True,
        )
        self.conv_2 = nn.Conv1d(
            int(config.embd_size / 2),
            config.channels,
            config.window_size,
            stride=config.window_size,
            bias=True,
        )
        self.pooling = nn.MaxPool1d(int(config.max_length / config.window_size))
        self.mlp = ClassificationHead(
            config.channels,
            config.num_labels,
            config.hidden_size,
            "leaky_relu",
            0.5,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> SequenceClassifierOutput:
        x = self.embed(input_ids)
        x = torch.transpose(x, -1, -2)
        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = F.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        x = cnn_value * gating_weight
        x = self.pooling(x)
        x = x.view(-1, self.config.channels)
        x = self.mlp(x)

        logits = x

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden_size: int = -1,
        hidden_act: Literal["tanh", "relu", "leaky_relu"] = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_act == "leaky_relu":
            activation = nn.LeakyReLU()
        elif hidden_act == "tanh":
            activation = nn.Tanh()
        elif hidden_act == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {hidden_act=}")

        if hidden_size > 0:
            self.mlp = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.Dropout(dropout),
                activation,
                nn.Linear(hidden_size, out_features),
            )
        else:
            self.mlp = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


def test():
    # pylint: disable=import-outside-toplevel
    # pylint: disable=wrong-import-position
    from functools import partial

    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    from src.data.loaders import get_bodmas_dataset
    from src.learn.utils import (
        get_tokenizer_object,
        get_fast_tokenizer,
        tokenize_fn,
        examples_to_text,
    )

    MAX_LENGTH = 2**16

    dataset, _ = get_bodmas_dataset()
    dataset = dataset.rename_column("labels", "label")
    # dataset["tr"] = dataset["tr"].select(list(range(128)))
    # dataset["vl"] = dataset["vl"].select(list(range(128)))
    # dataset["ts"] = dataset["ts"].select(list(range(128)))
    dataset["tr"] = dataset["tr"].select(range(256))
    dataset["vl"] = dataset["tr"].select(range(256))
    dataset.pop("ts")

    print(dataset)

    tokenizer = get_tokenizer_object()
    tokenizer = get_fast_tokenizer(tokenizer, model_max_length=MAX_LENGTH)

    dataset = dataset.map(partial(examples_to_text, max_length=MAX_LENGTH), batched=True)
    dataset = dataset.map(
        partial(
            tokenize_fn,
            tokenizer,
            truncation=True,
            max_length=MAX_LENGTH,
            return_overflowing_tokens=False,
        ),
        batched=True,
        remove_columns=["name", "bytes", "size", "length", "text"],
    )

    # config = MalConvGCTConfig(
    #     num_embd=len(tokenizer),
    #     out_size=dataset["tr"].info.features["label"].num_classes,
    #     pad_idx=tokenizer.pad_token_id,
    #     channels=128,
    #     window_size=512,
    #     stride=512,
    # )
    # model = MalConvGCT(config)

    # config = MalConvConfig(
    #     num_embd=len(tokenizer),
    #     out_size=dataset["tr"].info.features["label"].num_classes,
    #     pad_idx=tokenizer.pad_token_id,
    #     channels=128,
    #     window_size=64,
    #     stride=64,
    # )
    # model = MalConv(config)

    config = MyMalConvConfig(
        num_embd=len(tokenizer),
        out_size=dataset["tr"].info.features["label"].num_classes,
        pad_idx=tokenizer.pad_token_id,
        max_length=MAX_LENGTH,
        channels=64,
        # hidden_size=1024,
        # stride=320,
        # window_size=356,
    )
    model = MyMalConv(config)
    print(model)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        "./tmp/malconv_with_trainer",
        num_train_epochs=1,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=5e-4,
        # fp16=True,
        # no_cuda=True,
        logging_steps=1,
    )
    trainer = Trainer(
        model,
        args,
        data_collator,
        train_dataset=dataset["tr"],
        eval_dataset=dataset["vl"],
    )

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    test()
