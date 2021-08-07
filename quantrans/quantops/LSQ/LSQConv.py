import math
from enum import Enum

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from ..builder import QUANLAYERS
import torch.nn.functional as F
class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

@QUANLAYERS.register_module()
class LSQConv2d(nn.Conv2d):
    """Generates quantized convolutional layers.

args:
    nbits_w(int): bitwidth for the weight quantization,
    nbits_a(int): bitwidth for the activation quantization,
procedure:
    1.the quantizd representation of the data v_q can be calculated by follows:
    :math:`v_q = round(clip(\\frac{v}{s}, -Q_N, Q_P))`,
    where the scale s is learnable parameter.
    
    2. besides, the hypeparameter g can be defined by :math:`\\frac{1.0}{\\sqrt(x.numel() \\times Q_P)}`

    details can see in https://arxiv.org/pdf/1902.08153.pdf
"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        nbits_w=4,
        nbits_a=4,
        signed=False,
        quant_activation = True,
        q_mode=Qmodes.layer_wise,
        debug = False
    ):
        super(LSQConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.layer_type = 'LSQConv2d'
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.quant_activation = quant_activation
        self.q_mode = q_mode
        if self.q_mode == Qmodes.kernel_wise:
            self.weight_alpha = Parameter(torch.Tensor(out_channels))
        else:
            self.weight_alpha = Parameter(torch.Tensor(1))
        self.register_buffer("weight_init_state", torch.zeros(1))
        self.signed = signed
        self.act_alpha = Parameter(torch.Tensor(1))
        self.register_buffer("act_init_state", torch.zeros(1))
        self.debug = debug

    def lsq_quantization_act(self, x):
        assert self.act_alpha is not None
        if self.signed:
            Qn = -(2 ** (self.nbits_a - 1))
            Qp = 2 ** (self.nbits_a - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits_a - 1
        if self.training and self.act_init_state == 0:
            self.act_alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.act_init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)
        
        act_alpha = grad_scale(self.act_alpha, g)
        x_q = round_pass((x / act_alpha ).clamp(Qn, Qp)) * act_alpha 
        return x_q

    def lsq_quantization_weight(self, weight):
        Qn = -(2 ** (self.nbits_w - 1))
        Qp = 2 ** (self.nbits_w - 1) - 1
        if self.training and self.weight_init_state == 0:
            self.weight_alpha.data.copy_(2 * weight.abs().mean() / math.sqrt(Qp))
            self.weight_init_state.fill_(1)
        g = 1.0 / math.sqrt(weight.numel() * Qp)
        weight_alpha = grad_scale(self.weight_alpha, g)
        w_q = round_pass((weight / weight_alpha).clamp(Qn, Qp)) * weight_alpha
        return w_q    

    def forward(self, x):
        # if self.alpha is None:
        #     return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        w_q = self.lsq_quantization_weight(self.weight)
    
        if self.quant_activation:
            x_q = self.lsq_quantization_act(x)    
        else: 
            x_q = x
        if self.debug:
            self.Qweight = w_q.clone().detach()
            self.Qactivation = x_q.clone().detach()
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
