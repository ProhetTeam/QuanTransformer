import math
from enum import Enum

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from quantrans.builder import QUANLAYERS
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

class LSQPlus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, beta, lower_bound, upper_bound):
        r"""
        LSQPlus: y = Round[clamp((x - beta)/s, n, p)] * s + beta
        Args:
            x: input tensor
            scale: QAT scale
            beta: QAT offset
        return:
            x_quant: quantization tensor of x
        """
        x_hat = ((x - beta) / scale).round()
        ctx.save_for_backward(x, x_hat, scale, beta)
        ctx.constant = [lower_bound, upper_bound]
        x_hat = x_hat.clamp(lower_bound, upper_bound)
        x_quant = x_hat * scale + beta
        return x_quant
    
    @staticmethod
    def backward(ctx, grad_output):
        r"""
        Backward:
            x_gradient: x[x > p or x < n] = 0 else 1
            scale_gradient: -(x - beta)/s + round[(x - beta)/s] else n or p
            beta_gradient: 0 or 1
        """
        x, x_hat, scale, beta = ctx.saved_variables
        lower_bound, upper_bound = ctx.constant

        idx_smaller = (((x - beta)/scale).round() < lower_bound)
        idx_bigger = (((x - beta)/scale).round() > upper_bound)
        g_scale = 1.0 / math.sqrt(x.numel() * upper_bound) 
        g_offset =  min(1.0, 1.0 / math.sqrt(idx_smaller.sum() + idx_bigger.sum() + 1e-6))
        #g_offset = 1.0 / math.sqrt(x.numel() * upper_bound) 

        r"""1. input gradient"""
        x_grad = torch.ones_like(grad_output)
        x_grad[idx_smaller] = 0
        x_grad[idx_bigger] = 0
        x_grad *= grad_output

        r"""2. scale gradient"""
        scale_grad = -(x - beta)/scale + x_hat
        scale_grad[idx_smaller] = float(lower_bound)
        scale_grad[idx_bigger] = float(upper_bound)
        scale_grad = (scale_grad * grad_output).sum().unsqueeze(dim=0) * g_scale

        r"""3. offset gradient"""
        beta_grad = torch.zeros_like(x)
        beta_grad[idx_smaller] = 1
        beta_grad[idx_bigger] = 1
        beta_grad = (beta_grad * grad_output).sum().unsqueeze(dim=0) * g_offset
        return x_grad, scale_grad, beta_grad, None, None

@QUANLAYERS.register_module()
class LSQDPlusConv2d(nn.Conv2d):
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
        init_method_w = 1,
        customer_backward_w = False,
        add_offset_w = False,
        wquant_error_loss = False,
        nbits_a=4,
        init_method_a = 1,
        customer_backward_a = False,
        add_offset_a = True,
        aquant_error_loss = False,
        signed=False,
        auto_signed = False,
        quant_activation = True,
        momentum = 0.1,
        q_mode=Qmodes.layer_wise,
        quant_loss = False,
        scale_grad = True,
        debug = False
    ):
        super(LSQDPlusConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.layer_type = 'LSQDPlusConv2d'
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
        self.add_offset_a = add_offset_a
        self.add_offset_w = add_offset_w
        
        if self.q_mode == Qmodes.kernel_wise:
            self.weight_offset = Parameter(torch.Tensor(out_channels))
            self.act_offset = Parameter(torch.Tensor(out_channels))
        else:
            self.weight_offset = Parameter(torch.Tensor(1))
            self.act_offset = Parameter(torch.tensor(0.))
            self.register_buffer("running_weight_alpha", torch.zeros(1))
            self.register_buffer("running_act_alpha", torch.zeros(1))
            self.register_buffer("running_weight_offset", torch.zeros(1))
            self.register_buffer("running_act_offset", torch.zeros(1))
        self.momentum = momentum
        self.scale_grad = scale_grad
        self.debug = debug
        self.customer_backward_w = customer_backward_w
        self.customer_backward_a = customer_backward_a
        self.init_method_w = init_method_w
        self.init_method_a = init_method_a
        self.auto_signed = auto_signed
        self.wquant_error_loss = wquant_error_loss
        self.aquant_error_loss = aquant_error_loss
        self.quant_loss = quant_loss
    
    def _scale_offset_init(self, data: torch.Tensor, Qn, Qp, add_offset):
        target = data.detach().clone()
        beta = 0.1
        iters = 200

        alpha = Parameter((target.max() - target.min()) / (Qp - Qn) * 0.9)
        offset = Parameter(torch.tensor(0.))
        optimizer = torch.optim.SGD([alpha, offset], lr=0.01)
        dist = self.dist(data)
        for _ in range(iters):
            r"""
            g = 1.0 / math.sqrt(target.numel() * Qp)
            g_offset = grad_scale(offset, g)
            g_alpha = grad_scale(alpha, g)
            """
            if add_offset:
                target_quant = ((target - offset)/alpha).clamp(Qn, Qp).round() * alpha + offset
            else:
                target_quant = (target/alpha).clamp(Qn, Qp).round() * alpha
            weight = torch.exp(target.abs() * beta + dist)
            loss = ((target - target_quant)* weight).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return alpha.detach().clone(), offset.detach().clone()

    def lsq_quantization_act(self, x):
        assert self.act_alpha is not None
        if self.signed or(self.auto_signed and (True in (x < 0))):
            if self.training and self.act_init_state == 0: 
                self.signed = True
            Qn = -(2 ** (self.nbits_a - 1))
            Qp = 2 ** (self.nbits_a - 1) - 1
        else:
            if self.training and self.act_init_state == 0: 
                self.signed = False
            Qn = 0
            Qp = 2 ** self.nbits_a - 1
        if self.training and self.act_init_state == 0:
            if self.init_method_a == 1:
                self.act_alpha.data.copy_((x.max() - x.min()) / (Qp - Qn) * 0.9)
            elif self.init_method_a == 2:
                self.act_alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            elif self.init_method_a == 3:
                scale_temp, offset_temp = self._scale_offset_init(x, Qn, Qp, self.add_offset_a)
                self.act_alpha.data.copy_(scale_temp)
            else:
                raise NotImplementedError
            self.running_act_alpha = self.act_alpha.detach().clone()
            if self.add_offset_a:
                if self.init_method_a == 3:
                    self.act_offset.data.copy_(offset_temp)
                else:
                    self.act_offset.data.copy_(x.min() * 0.9 - Qn * self.act_alpha)
                self.running_act_offset = self.act_offset.data
                #self.act_offset.data.copy_(x.mean()) 
            self.act_init_state.fill_(1)
        elif self.training:
            self.running_act_alpha += \
                self.momentum * (self.act_alpha - self.running_act_alpha).data
            self.running_weight_offset += \
                self.momentum * (self.act_offset - self.running_act_offset).data

        if not self.customer_backward_a:
            g = 1.0 / math.sqrt(x.numel() * Qp) if self.scale_grad else 1.0
            act_alpha = grad_scale(self.act_alpha, g) if self.training else self.running_act_alpha
            if not self.add_offset_a:
                x_q = round_pass((x / act_alpha ).clamp(Qn, Qp)) * act_alpha 
            else:
                act_offset = grad_scale(self.act_offset, g) if self.training else self.running_act_offset
                x_q = round_pass(((x - act_offset) / act_alpha).clamp(Qn, Qp)) * act_alpha + act_offset
        else:
            x_q = LSQPlus.apply(x, self.act_alpha, self.act_offset, Qn, Qp)
        if self.aquant_error_loss:
            dist = self.dist(x) 
            self.act_quant_loss = 1.0 * ((x - x_q)* torch.exp(x.abs() * 0.1 + dist)).abs().mean()
        return x_q

    def lsq_quantization_weight(self, weight):
        Qn = -(2 ** (self.nbits_w - 1))
        Qp = 2 ** (self.nbits_w - 1) - 1
            
        if self.training and self.weight_init_state == 0:
            if self.init_method_w == 1:
                self.weight_alpha.data.copy_(torch.max( torch.abs(weight.mean() - 3 * weight.std()), 
                                        torch.abs(weight.mean() + 3 * weight.std())) / ((Qp - Qn)/2))
            elif self.init_method_w == 2:
                self.weight_alpha.data.copy_(2 * weight.abs().mean() / math.sqrt(Qp))
            elif self.init_method_w == 3:
                scale_temp, offset_temp = self._scale_offset_init(weight, Qn, Qp, self.add_offset_w)
                self.weight_alpha.data.copy_(scale_temp)
            else:
                raise NotImplementedError
            self.running_weight_alpha = self.weight_alpha.detach().clone()
            if self.add_offset_w:
                if self.init_method_w == 3:
                    self.weight_offset.data.copy_(offset_temp)
                else:
                    self.weight_offset.data.copy_(weight.mean())
                    self.running_weight_offset = self.weight_offset.data
            self.weight_init_state.fill_(1)
        elif self.training:
            self.running_weight_alpha += \
                self.momentum * (self.weight_alpha - self.running_weight_alpha).data
            self.running_weight_offset += \
                self.momentum * (self.weight_offset - self.running_weight_offset).data

        if not self.customer_backward_w:
            g = 1.0 / math.sqrt(weight.numel() * Qp) if self.scale_grad else 1.0
            weight_alpha = grad_scale(self.weight_alpha, g) if self.training else self.running_weight_alpha 
            if not self.add_offset_w:
                w_q = round_pass((weight / weight_alpha).clamp(Qn, Qp)) * weight_alpha
            else:
                weight_offset = grad_scale(self.weight_offset, g) if self.training else self.running_weight_offset
                w_q = round_pass(((weight - weight_offset) / weight_alpha).clamp(Qn, Qp)) * weight_alpha + weight_offset
        else:
            w_q = LSQPlus.apply(weight, self.weight_alpha, self.weight_offset, Qn, Qp)

        if self.wquant_error_loss:
            dist = self.dist(weight) 
            self.weight_quant_loss = 1.0 * ((weight - w_q)* torch.exp(weight.abs() * 0.1 + dist)).abs().mean()
        return w_q
    
    def dist(self, data, bins = 600):
        bin_size = (data.max() - data.min()) / bins
        hisc = data.detach().histc(bins = bins) / data.numel()
        xbin_idx = ((data - data.min()) / bin_size).floor().clamp(0, bins - 1)
        res = hisc[xbin_idx.long()]
        res = res/res.sum()
        return res

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
        res = F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.quant_loss:
            res_target = F.conv2d(x.detach().clone(), self.weight.detach().clone(), self.bias, self.stride, self.padding, self.dilation, self.groups)
            dist = self.dist(res_target) 
            self.quant_error_loss = 0.01 * ((res - res_target.detach()) * torch.exp(res_target.abs() * 0.1 + dist)).abs().mean()
        return res
