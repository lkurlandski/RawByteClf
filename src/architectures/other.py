"""
"""

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    """
    A wrapper around torchvision's sigmoid_focal_loss that applies an optional per-class weight.

    Arguments:
        alpha (float): Weighting factor for positive examples (usually between 0 and 1).
        gamma (float): Exponent of the modulating factor (1 - p_t). Typically set to 2.
        reduction (str): Specifies the reduction to apply: 'none', 'mean', 'sum'.
                         Default: 'mean'.
        weight (Tensor, optional): A tensor of shape [num_classes] to weight each class.
                                   Similar to BCELoss(weight=...).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float)
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, input, target):
        """
        Compute focal loss using sigmoid_focal_loss and apply class weights if provided.

        Arguments:
            input (Tensor): Logits of shape (N, C), where C is the number of classes.
            target (Tensor): Targets of the same shape as input, with values in {0, 1}.
        
        Returns:
            Tensor: The computed focal loss.
        """
        # Compute focal loss per-element without reduction
        focal_loss = sigmoid_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma, reduction='none')

        # If weight is provided, apply per-class weights
        if self.weight is not None:
            # focal_loss: [N, C]
            # weight: [C]
            # Broadcast multiplication to apply class weights
            focal_loss = focal_loss * self.weight

        # Apply the specified reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
