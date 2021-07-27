import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import QUANLAYERS

DEBUG = False
class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        delta = torch.max(x) - torch.min(x)
        x = (x/delta + 0.5)
        return x.round() * 2 - 1
    @staticmethod
    def backward(ctx, g):
        return g

@QUANLAYERS.register_module()
class DSQConvV2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1,
                num_bit_w = 8,
                num_bit_a = 8,
                QInput = True,
                bSetQ = True,
                alpha_thres = 0.5,
                bias_quant = False):
        super(DSQConvV2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit_w = num_bit_w
        self.num_bit_a = num_bit_a
        self.quan_input = QInput
        self.bit_range_w = 2**self.num_bit_w - 1
        self.bit_range_a = 2**self.num_bit_a - 1
        self.is_quan = bSetQ
        self.momentum = momentum
        self.alpha_thres = alpha_thres
        self.bias_quant = bias_quant
        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.uW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.lW  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
            self.register_buffer('running_uW', torch.tensor([self.uW.data])) # init with uw
            self.register_buffer('running_lW', torch.tensor([self.lW.data])) # init with lw
            self.alphaW = nn.Parameter(data = torch.tensor(0.2).float())
            # Bias
            if self.bias is not None:
                self.uB = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.lB  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
                self.register_buffer('running_uB', torch.tensor([self.uB.data]))# init with ub
                self.register_buffer('running_lB', torch.tensor([self.lB.data]))# init with lb
                self.alphaB = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.lA  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
                self.register_buffer('running_uA', torch.tensor([self.uA.data])) # init with uA
                self.register_buffer('running_lA', torch.tensor([self.lA.data])) # init with lA
                self.alphaA = nn.Parameter(data = torch.tensor(0.2).float())

    def clipping(self, x, upper, lower):
        x = x.clamp(lower.item(), upper.item())
        return x

    def floor_pass(self, x):
        y = torch.floor(x)
        y_grad = x
        return y.detach() - y_grad.detach() + y_grad

    def phi_function(self, x, mi, alpha, delta):
        alpha = alpha.clamp(1e-6, self.alpha_thres - 1e-6)
        s = 1/(1-alpha)
        k = (2/alpha - 1).log() * (1/delta)
        x = (((x - mi) *k ).tanh()) * s
        return x

    def sgn(self, x):
        x = RoundWithGradient.apply(x)
        return x

    def dequantize(self, x, lower_bound, delta, interval):

        # save mem
        x =  ((x+1)/2 + interval) * delta + lower_bound

        return x

    def forward(self, x):
        if self.is_quan:
            if torch.equal(self.lW, torch.tensor(-1. * 2**32, device = self.lW.device)) and \
                torch.equal(self.uW, torch.tensor(1. *2**31 - 1, device = self.uW.device)):
                self.lW = nn.Parameter(torch.min(self.weight))
                self.uW =  nn.Parameter(torch.max(self.weight))
                self.running_lW = torch.min(self.weight)
                self.running_uW = torch.max(self.weight)

            if self.training:
                self.running_uW = self.running_uW * (1-self.momentum) + self.momentum * self.uW
                self.running_lW = self.running_lW * (1-self.momentum) + self.momentum * self.lW
                Qweight = self.clipping(self.weight, self.uW, self.lW)
            else:
                Qweight = self.clipping(self.weight, self.running_uW, self.running_lW)
            cur_max = self.uW if self.training else self.running_uW
            cur_min = self.lW if self.training else self.running_lW

            delta =  (cur_max - cur_min)/(self.bit_range_w)
            #interval = (Qweight - cur_min) //delta ## not differential ??
            interval = self.floor_pass((Qweight - cur_min) /delta)
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.alphaW, delta)
            Qweight = self.sgn(Qweight)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval)

            Qbias = self.bias

            if self.bias is not None and self.bias_quant:
                if torch.equal(self.lB, torch.tensor(-1. * 2**32, device = self.lB.device)) and \
                    torch.equal(self.uB, torch.tensor(1. *2**31 - 1, device = self.uB.device)):
                    self.lB = nn.Parameter(torch.min(self.bias))
                    self.uB =  nn.Parameter(torch.max(self.bias))
                    self.running_lB = torch.min(self.bias)
                    self.running_uB = torch.max(self.bias)

                if self.training:
                    self.running_uB = self.running_uB * (1-self.momentum) + self.momentum * self.uB.clone()
                    self.running_lB = self.running_lB * (1-self.momentum) + self.momentum * self.lB.clone()
                    Qbias = self.clipping(self.bias, self.uB, self.lB)
                else:
                    Qbias = self.clipping(self.bias, self.running_uB, self.running_lB)

                cur_max = self.uB if self.training else self.running_uB
                cur_min = self.lB if self.training else self.running_lB

                delta =  (cur_max - cur_min)/(self.bit_range_w)
                #interval = (Qbias - cur_min) //delta
                interval = self.floor_pass((Qbias - cur_min) /delta)
                mi = (interval + 0.5) * delta + cur_min
                Qbias = self.phi_function(Qbias, mi, self.alphaB, delta)
                Qbias = self.sgn(Qbias)
                Qbias = self.dequantize(Qbias, cur_min, delta, interval)

            # Input(Activation)
            Qactivation = x
            if self.quan_input:
                if torch.equal(self.lA, torch.tensor(-1. * 2**32, device = self.lA.device)) and \
                    torch.equal(self.uA, torch.tensor(1. *2**31 - 1, device = self.uA.device)):
                    self.lA = nn.Parameter(torch.min(x))
                    self.uA =  nn.Parameter(torch.max(x))
                    self.running_lA = torch.min(x)
                    self.running_uA = torch.max(x)

                if self.training:
                    self.running_uA = self.running_uA * (1-self.momentum) + self.momentum * self.uA.clone()
                    self.running_lA = self.running_lA * (1-self.momentum) + self.momentum * self.lA.clone()
                    Qactivation = self.clipping(x, self.uA, self.lA)
                else:
                    Qactivation = self.clipping(x, self.running_uA, self.running_lA)

                cur_max = self.uA if self.training else self.running_uA
                cur_min = self.lA if self.training else self.running_lA

                delta =  (cur_max - cur_min)/(self.bit_range_a)
                #interval = (Qactivation - cur_min) //delta
                interval = self.floor_pass((Qactivation - cur_min) /delta)
                mi = (interval + 0.5) * delta + cur_min
                Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                Qactivation = self.sgn(Qactivation)
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)

            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output
