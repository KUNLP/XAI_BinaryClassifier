# https://github.com/KLUE-benchmark/KLUE-baseline/blob/8a03c9447e4c225e806877a84242aea11258c790/klue_baseline/models/dependency_parsing.py
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class BiAttention(nn.Module):
    def __init__(
        self,
        input_size_encoder,
        input_size_decoder,
        num_labels,
        biaffine=True,
        **kwargs
    ):
        super(BiAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(
                torch.Tensor(
                    self.num_labels, self.input_size_decoder, self.input_size_encoder
                )
            )
        else:
            self.register_parameter("U", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_d)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(self, input_e, input_d, mask_d=None, mask_e=None):
        assert input_d.size(0) == input_e.size(0)
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # input_d : [b, t, d]
        # input_e : [b, s, e]
        # out_d : [b, l, d, 1]
        # out_e : [b, l ,1, e]
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        if self.biaffine:
            # output : [b, 1, t, d] * [l, d, e] -> [b, l, t, e]
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            # output : [b, l, t, e] * [b, 1, e, s] -> [b, l, t, s]
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))
            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = (
                output
                * mask_d.unsqueeze(1).unsqueeze(3)
                * mask_e.unsqueeze(1).unsqueeze(2)
            )

        # input1 = (batch_size, input11, input12)
        # input2 = (batch_size, input21, input22)
        return output # (batch_size, output_size, input11, input21)

class BiLinear(nn.Module):
    def __init__(self, left_features: int, right_features: int, out_features: int):
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.right_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left: torch.Tensor, input_right: torch.Tensor) -> torch.Tensor:
        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], "batch size of left and right inputs mis-match: (%s, %s)" % (
            left_size[:-1],
            right_size[:-1],
        )
        batch = int(np.prod(left_size[:-1]))

        input_left = input_left.contiguous().view(batch, self.left_features)
        input_right = input_right.contiguous().view(batch, self.right_features)

        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
        return output.view(left_size[:-1] + (self.out_features,))
