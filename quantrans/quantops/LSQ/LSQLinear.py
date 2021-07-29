import math
from enum import Enum

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from ..builder import QUANLAYERS
import torch.nn.functional as F


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


def lsq_quantization_act(x, nbits, alpha, training, init_state, signed=False):
    assert alpha is not None
    if signed:
        Qn = -(2 ** (nbits - 1))
        Qp = 2 ** (nbits - 1) - 1
    else:
        Qn = 0
        Qp = 2 ** nbits - 1
    if training and init_state == 0:
        alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
        init_state.fill_(1)

    g = 1.0 / math.sqrt(x.numel() * Qp)
    
    alpha = grad_scale(alpha, g)
    x_q = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
    return x_q

def lsq_quantization_linear(weight , nbits, alpha, training, init_state):
    Qn = -(2 ** (nbits - 1))
    Qp = 2 ** (nbits - 1) - 1
    if training and init_state == 0:
        alpha.data.copy_(2 * weight.abs().mean() / math.sqrt(Qp))
        init_state.fill_(1)
    g = 1.0 / math.sqrt(weight.numel() * Qp)
    alpha = grad_scale(alpha, g)
    l_q = round_pass((weight / alpha).clamp(Qn, Qp)) * alpha
    return l_q




@QUANLAYERS.register_module()
class LSQLinear(nn.Linear):
    """Generates quantized linear layers.

Same as LSQConv()

"""
    def __init__(self, in_features, out_features, bias=True, nbits_w=8, nbits_a=8, signed=False):
        super(LSQLinear, self).__init__(in_features, out_features, bias=bias)
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.signed = signed
        self.linear_alpha = Parameter(torch.Tensor(1))
        self.act_alpha = Parameter(torch.Tensor(1))
        self.register_buffer("linear_init_state", torch.zeros(1))
        #self.training = training
        self.register_buffer("act_init_state", torch.zeros(1))

    def forward(self, x):
        # if self.alpha is None:
        #     return F.linear(x, self.weight, self.bias)
        
        l_q = lsq_quantization_linear(self.weight, self.nbits_w, self.linear_alpha,self.training,self.linear_init_state)
        x_q = lsq_quantization_act(x, self.nbits_a, self.act_alpha, self.training,self.act_init_state, self.signed)
        
        # Method1:
        return F.linear(x_q, l_q, self.bias)
