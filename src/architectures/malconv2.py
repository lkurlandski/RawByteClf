"""
Implementation of the MalConv2 & MalConvGCG architectures from:
  @inproceedings{raff2021classifying,
    title={Classifying sequences of extreme length with constant memory applied to malware detection},
    author={Raff, Edward and Fleshman, William and Zak, Richard and Anderson, Hyrum S and Filar, Bobby and McLean, Mark},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2021}
  }
"""

import os
import sys
from typing import Literal, Optional
import warnings

import numpy as np
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.nn import functional as F

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.architectures.ensemble import EnsembleForSequenceClassification


################################################################################
########################### Original Implementation ############################
################################################################################


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)

    raise RuntimeError("Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


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


def drop_zeros_hook(module, grad_input, grad_out):  # pylint: disable=unused-argument
    """
    This function is used to replace gradients that are all zeros with None
    In pyTorch None will not get back-propogated
    So we use this as a approximation to saprse BP to avoid redundant and useless work
    """
    grads = []
    with torch.no_grad():
        for g in grad_input:
            if torch.nonzero(g).shape[0] == 0:  # ITS ALL EMPTY!
                grads.append(g.to_sparse())
            else:
                grads.append(g)

    return tuple(grads)


class CatMod(torch.nn.Module):
    def __init__(self):
        super(CatMod, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=2)


class LowMemConvBase(nn.Module):
    def __init__(self, chunk_size: int = 65536, overlap: int = 512, min_chunk_size: int = 1024, pad_token_id: int = 0):
        """
        Args:
          chunk_size: how many bytes at a time to process. Increasing may improve compute efficent,
           but use more memory. Total memory use will be a function of chunk_size, and not of the
           length of the input sequence L.
          overlap: how many bytes of overlap to use between chunks
        """
        super(LowMemConvBase, self).__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.pad_token_id = pad_token_id

        # Used for pooling over time in a meory efficent way
        self.pooling = nn.AdaptiveMaxPool1d(1)
        # self.pooling.register_backward_hook(drop_zeros_hook)
        self.cat = CatMod()
        self.cat.register_backward_hook(drop_zeros_hook)
        self.receptive_field = None

        # Used to force checkpoint code to behave correctly due to poor design:
        # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def processRange(self, x, **kwargs) -> Tensor:
        """
        This method does the work to convert an LongTensor input x of shape (B, L),
          where B is the batch size and L is the length of the input. The output of this function
          should be a tensor of (B, C, L), where C is the number of channels, and L is again the
          input length (though its OK if it got a little shorter due to convs without padding or something).
        """
        ...

    def determinRF(self):
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
                except Exception:
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

    def pool_group(self, *args):
        x = self.cat(args)
        x = self.pooling(x)
        return x

    def seq2fix(self, x, pr_args={}):
        """
        Takes in an input LongTensor of (B, L) that will be converted to a fixed length representation (B, C),
         where C is the number of channels provided by the base_network  given at construction.
        """

        receptive_window, stride, out_channels = self.determinRF()

        if x.shape[1] < receptive_window:
            x = F.pad(x, (0, receptive_window - x.shape[1]), value=self.pad_token_id)

        batch_size = x.shape[0]
        length = x.shape[1]

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

        final_indices = [np.unique(winner_indices[b, :]) for b in range(batch_size)]
        chunk_list = [
            [x[b : b + 1, max(i - receptive_window, 0) : min(i + receptive_window, length)] for i in final_indices[b]]
            for b in range(batch_size)
        ]
        chunk_list = [torch.cat(c, dim=1)[0, :] for c in chunk_list]
        x_selected = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)
        if cur_device is not None:
            x_selected = x_selected.to(cur_device)
        x_selected = self.processRange(x_selected.long(), **pr_args)
        x_selected = self.pooling(x_selected)
        x_selected = x_selected.view(x_selected.size(0), -1)

        return x_selected


class MalConvML(LowMemConvBase):

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        channels: int,
        stride: int,
        kernel_size: int,
        out_size: int,
        layers: int = 1,
        pad_token_id: int = 0,
    ) -> None:
        super(MalConvML, self).__init__(pad_token_id=pad_token_id)
        self.embd = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_size, channels * 2, kernel_size, stride=stride, bias=True)]
            + [nn.Conv1d(channels, channels * 2, kernel_size, stride=1, bias=True) for _ in range(layers - 1)]
        )
        self.convs_1 = nn.ModuleList([nn.Conv1d(channels, channels, 1, bias=True) for _ in range(layers)])
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)

    def processRange(self, x):
        x = self.embd(x)
        x = x.permute(0, 2, 1).contiguous()
        for conv_glu, conv_share in zip(self.convs, self.convs_1):
            x = F.leaky_relu(conv_share(F.glu(conv_glu(x.contiguous()), dim=1)))
        return x

    def forward(self, x):
        post_conv = x = self.seq2fix(x)
        penult = x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x, penult, post_conv


class MalConvGCT(LowMemConvBase):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        channels: int,
        stride: int,
        kernel_size: int,
        out_size: int,
        layers: int = 1,
        pad_token_id: int = 0,
    ) -> None:
        super(MalConvGCT, self).__init__(pad_token_id=pad_token_id)
        self.embd = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        self.context_net = MalConvML(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            channels=channels,
            stride=stride,
            kernel_size=kernel_size,
            out_size=channels,
            layers=layers,
            pad_token_id=pad_token_id,
        )
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_size, channels * 2, kernel_size, stride=stride, bias=True)]
            + [nn.Conv1d(channels, channels * 2, kernel_size, stride=1, bias=True) for _ in range(layers - 1)]
        )
        self.linear_atn = nn.ModuleList([nn.Linear(channels, channels) for _ in range(layers)])
        self.convs_share = nn.ModuleList([nn.Conv1d(channels, channels, 1, bias=True) for _ in range(layers)])
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)

    def determinRF(self):
        """
        Over-write the determinRF call to use the base context_net to detemrin RF.
         We should have the same totla RF, and this will simplify logic significantly.
        """
        return self.context_net.determinRF()

    def processRange(self, x, gct=None):
        if gct is None:
            raise Exception("No Global Context Given")

        x = self.embd(x)
        x = x.permute(0, 2, 1)

        for conv_glu, linear_cntx, conv_share in zip(self.convs, self.linear_atn, self.convs_share):
            x = F.glu(conv_glu(x), dim=1)
            x = F.leaky_relu(conv_share(x))
            x_len = x.shape[2]
            B = x.shape[0]
            C = x.shape[1]

            sqrt_dim = np.sqrt(x.shape[1])
            # we are going to need a version of GCT with a time dimension, which
              # we will adapt as needed to the right length
            ctnx = torch.tanh(linear_cntx(gct))

            # Size is (B, C), but we need (B, C, 1) to use as a 1d conv filter
            ctnx = torch.unsqueeze(ctnx, dim=2)
            # roll the batches into the channels
            x_tmp = x.view(1, B * C, -1)
            # Now we can apply a conv with B groups, so that each batch gets its
              # own context applied only to what was needed
            x_tmp = F.conv1d(x_tmp, ctnx, groups=B)
            # x_tmp will have a shape of (1, B, L), now we just need to re-order
              # the data back to (B, 1, L)
            x_gates = x_tmp.view(B, 1, -1)

            # Now we effectively apply σ(x_t^T tanh(W c))
            gates = torch.sigmoid(x_gates)
            x = x * gates

        return x

    def forward(self, x):
        global_context = CheckpointFunction.apply(self.context_net.seq2fix, 1, x)
        post_conv = x = self.seq2fix(x, pr_args={"gct": global_context})
        penult = x = F.leaky_relu(self.fc_1(x))
        x = self.fc_2(x)
        return x, penult, post_conv


################################################################################
################################ Wrap for HF ###################################
################################################################################


class MalConv2Config(PretrainedConfig):

    """
    Configuration used by original authors:

        >>> # For MalConv2
        >>> MalConv2Config(
                vocab_size=257,
                embedding_size=8,
                channels=128,
                stride=512,
                kernel_size=512,
                layers=1,
                pad_token_id=0,
            )
        >>> # For MalConvGCG
        >>> MalConv2Config(
                vocab_size=257,
                embedding_size=8,
                channels=256,
                stride=64,
                kernel_size=256,
                layers=1,
                pad_token_id=0,
            )
    """

    def __init__(
        self,
        mode: Literal["gcg", "base"] = "gcg",
        vocab_size: int = 264,
        embedding_size: int = 256,
        pad_token_id: int = 0,
        channels: int = 128,
        stride: int = 512,
        kernel_size: int = 512,
        layers: int = 1,
        head_hidden_size: int = 0,
        head_num_hidden_layers: int = 0,
        head_dropout: float = 0.1,
        **kwds,
    ) -> None:
        self.mode = mode
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pad_token_id = pad_token_id
        self.channels = channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.layers = layers
        self.head_hidden_size = head_hidden_size
        self.head_num_hidden_layers = head_num_hidden_layers
        self.head_dropout = head_dropout
        if head_hidden_size != 0 or head_num_hidden_layers != 0 or head_dropout != 0.1:
            raise NotImplementedError("I never actually implemented this!")
        super().__init__(**kwds)

    @property
    def hidden_size(self) -> int:
        return self.channels


class MalConv2PreTrainedModel(PreTrainedModel):
    config_class = MalConv2Config
    base_model_prefix = "malconv"
    supports_gradient_checkpointing = False


class MalConv2(MalConv2PreTrainedModel):

    def __init__(self, config: MalConv2Config) -> None:
        super().__init__(config)
        kwds = {
            "vocab_size": config.vocab_size,
            "channels": config.channels,
            "kernel_size": config.kernel_size,
            "stride": config.stride,
            "layers": config.layers,
            "embedding_size": config.embedding_size,
            "pad_token_id": config.pad_token_id,
            "out_size": config.num_labels,
        }
        if config.mode == "gcg":
            self.malconv = MalConvGCT(**kwds)
        elif config.mode == "base":
            self.malconv = MalConvML(**kwds)

    def forward(self, input_ids: Tensor) -> BaseModelOutput:
        x, penult, post_conv = self.malconv(input_ids)
        return BaseModelOutput(last_hidden_state=x, hidden_states=(penult, post_conv))


class MalConv2ForSequenceClassification(MalConv2PreTrainedModel):

    def __init__(self, config: MalConv2Config) -> None:
        super().__init__(config)
        self.malconv = MalConv2(config)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> SequenceClassifierOutput:
        x: Tensor = self.malconv(input_ids)[0]

        logits = x
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class MalConv2EnsembleForSequenceClassification(EnsembleForSequenceClassification):

    def __init__(self, config: MalConv2Config) -> None:
        super().__init__(config, MalConv2)

    def get_pooled_hidden_states(self, backbone: MalConv2, input_ids: Tensor, **kwds) -> Tensor:  # pylint: disable=arguments-differ
        # This implementation uses the penultimate activations as pooled hidden states.
        output = backbone(input_ids=input_ids)
        pooled_hidden_states = output.hidden_states[0]
        return pooled_hidden_states


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MalConv2Config(
        mode="gcg",
        vocab_size=256,
        embedding_size=256,
        pad_token_id=0,
        channels=128,
        stride=256,
        kernel_size=512,
    )
    model = MalConv2ForSequenceClassification(config).to(device)

    length = 2**19 + 1
    bos = torch.tensor([1])
    eos = torch.tensor([2])
    x = torch.randint(3, config.vocab_size, (length - 2,))
    x = torch.cat([bos, x, eos], dim=0)
    x = x.unsqueeze(0).to(device)

    model(x)


if __name__ == "__main__":
    test()
