# -*- coding: utf-8 -*-
import torch
from torch import nn

### Currently DO NOT use
class MarginLoss(nn.Module):
    def __init__(self,):
        """

        :param d_model: embedding dim
        :param nhead:
        :param num_layers:
        """
        super().__init__()
        self.loss = nn.Margin

    def forward(self, inputs, mask=None):
        return self.transformer(src=inputs, mask=mask)