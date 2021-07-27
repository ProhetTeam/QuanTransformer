import math
from enum import Enum

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from quantrans.builder import QUANLAYERS
import torch.nn.functional as F


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


@QUANLAYERS.register_module()
class LSQPlusLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=8, nbits_a=8, signed=False, add_offset = True):
        super(LSQPlusLinear, self).__init__(in_features, out_features, bias=bias)
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.signed = signed
        self.linear_alpha = Parameter(torch.Tensor(1))
        self.act_alpha = Parameter(torch.Tensor(1))
        self.register_buffer("linear_init_state", torch.zeros(1))
        self.register_buffer("act_init_state", torch.zeros(1))
        self.add_offset = add_offset
        self.linear_offset = Parameter(torch.Tensor(1))
        self.act_offset = Parameter(torch.Tensor(1))

    def lsq_quantization_act(self, x):
        assert self.act_alpha is not None
        if self.signed:
            Qn = -(2 ** (self.nbits_a - 1))
            Qp = 2 ** (self.nbits_a - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits_a - 1
        if self.training and self.act_init_state == 0:
            ''' LSQ: Implementation
            self.act_alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            '''
            self.act_alpha.data.copy_((x.max() - x.min()) / (Qp - Qn) * 0.9)
            if self.add_offset:
                self.act_offset.data.copy_(x.min() * 0.9 - Qn * self.act_alpha)
            self.act_init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)
        act_alpha = grad_scale(self.act_alpha, g)
        if not self.add_offset:
            x_q = round_pass((x / act_alpha).clamp(Qn, Qp)) * act_alpha
        else:
            act_offset = grad_scale(self.act_offset, g)
            x_q = round_pass((x - act_offset) / act_alpha).clamp(Qn, Qp) * act_alpha + act_offset
        return x_q

    def lsq_quantization_linear(self, weight):
        Qn = -(2 ** (self.nbits_w - 1))
        Qp = 2 ** (self.nbits_w - 1) - 1
        if self.training and self.linear_init_state == 0:
            ''' LSQ: Implementation
            self.linear_alpha.data.copy_(2 * weight.abs().mean() / math.sqrt(Qp))
            '''
            self.linear_alpha.data.copy_(torch.max( torch.abs(weight.mean() - 3 * weight.std()), 
                                         torch.abs(weight.mean() + 3 * weight.std())) / ((Qp - Qn)/2))
            if self.add_offset:
                self.linear_offset.data.copy_(weight.mean())
            self.linear_init_state.fill_(1)
        g = 1.0 / math.sqrt(weight.numel() * Qp)
        weight_alpha = grad_scale(self.linear_alpha, g)
        if not self.add_offset:
            l_q = round_pass((weight / weight_alpha).clamp(Qn, Qp)) * weight_alpha
        else:
            linear_offset = grad_scale(self.linear_offset, g)
            l_q = round_pass((weight - linear_offset) / weight_alpha).clamp(Qn, Qp) * weight_alpha + linear_offset
        return l_q

    def forward(self, x):
        # if self.alpha is None:
        #     return F.linear(x, self.weight, self.bias)
        
        l_q = self.lsq_quantization_linear(self.weight)
        x_q = self.lsq_quantization_act(x)
        
        # Method1:
        return F.linear(x_q, l_q, self.bias)
