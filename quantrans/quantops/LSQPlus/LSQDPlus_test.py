import torch
import torch.nn as nn
import random
import numpy as np
# from quantops.LSQPlus import LSQDPlusConv2d

r"""test """
class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min = 0, max = 1):
        ctx.save_for_backward(x)
        ctx.constant = [min, max]
        return x.clamp(min, max)
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        min, max = ctx.constant 
        grad_input = g.clone()
        grad_input[x < min] = 0
        grad_input[x > max] = 0
        return grad_input, None, None

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

import math
from enum import Enum
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
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
        #g_offset =  min(1.0, 1.0 / math.sqrt(idx_smaller.sum() + idx_bigger.sum() + 1e-6))
        g_offset = 1.0 / math.sqrt(x.numel() * upper_bound) 

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
        nbits_a=4,
        init_method_a = 1,
        customer_backward_a = False,
        signed=False,
        quant_activation = True,
        add_offset = True,
        momentum = 0.1,
        q_mode=Qmodes.layer_wise,
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
        self.add_offset = add_offset
        
        if self.q_mode == Qmodes.kernel_wise:
            self.weight_offset = Parameter(torch.Tensor(out_channels))
            self.act_offset = Parameter(torch.Tensor(out_channels))
        else:
            self.weight_offset = Parameter(torch.Tensor(1))
            self.act_offset = Parameter(torch.Tensor(1))
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

    def lsq_quantization_act(self, x):
        assert self.act_alpha is not None
        if self.signed:
            Qn = -(2 ** (self.nbits_a - 1))
            Qp = 2 ** (self.nbits_a - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits_a - 1
        if self.training and self.act_init_state == 0:
            if self.init_method_a == 1:
                self.act_alpha.data.copy_((x.max() - x.min()) / (Qp - Qn) * 0.9)
            elif self.init_method_a == 2:
                self.act_alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            else:
                raise NotImplementedError
            self.running_act_alpha = self.act_alpha.detach().clone()
            if self.add_offset:
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
            if not self.add_offset:
                x_q = round_pass((x / act_alpha ).clamp(Qn, Qp)) * act_alpha 
            else:
                act_offset = grad_scale(self.act_offset, g) if self.training else self.running_act_offset
                x_q = round_pass(((x - act_offset) / act_alpha).clamp(Qn, Qp)) * act_alpha + act_offset
        else:
            #x_q = LSQPlus.apply(x, self.act_alpha, self.act_offset, Qn, Qp)
            g = 1.0 / math.sqrt(x.numel() * Qp) if self.scale_grad else 1.0
            act_alpha = grad_scale(self.act_alpha, g) if self.training else self.running_act_alpha
            act_offset = grad_scale(self.act_offset, g) if self.training else self.running_act_offset
            x_q =  Round.apply(Clamp.apply((x - act_offset) / act_alpha, Qn, Qp)) * act_alpha + act_offset
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
            else:
                raise NotImplementedError
            self.running_weight_alpha = self.weight_alpha.detach().clone()
            if self.add_offset:
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
            if not self.add_offset:
                w_q = round_pass((weight / weight_alpha).clamp(Qn, Qp)) * weight_alpha
            else:
                weight_offset = grad_scale(self.weight_offset, g) if self.training else self.running_weight_offset
                w_q = round_pass(((weight - weight_offset) / weight_alpha).clamp(Qn, Qp)) * weight_alpha + weight_offset
        else:
            #w_q = LSQPlus.apply(weight, self.weight_alpha, self.weight_offset, Qn, Qp)
            g = 1.0 / math.sqrt(weight.numel() * Qp) if self.scale_grad else 1.0
            weight_alpha = grad_scale(self.weight_alpha, g) if self.training else self.running_weight_alpha 
            weight_offset = grad_scale(self.weight_offset, g) if self.training else self.running_weight_offset
            w_q = Round.apply(Clamp.apply((weight - weight_offset) / weight_alpha, Qn, Qp)) * weight_alpha + weight_offset 
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

r"""test  end """

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

''' Define hook 
'''
# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.name = module._get_name()
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

class LSQDPlusTestModel1(nn.Module):
    def __init__(self):
        super(LSQDPlusTestModel1, self).__init__()
        
        torch.manual_seed(seed)
        self.conv1 = LSQDPlusConv2d(
                3, 8, 3, 1, 1, bias= None,
                nbits_w=4,
                init_method_w = 2,
                customer_backward_w = True, 
                nbits_a=4,
                init_method_a = 2,
                customer_backward_a = True,
                add_offset = True, 
                momentum = 1.0, 
                debug = False)
        torch.manual_seed(seed)
        self.fc = nn.Linear(200, 10)
        self.grad_input_list = []
        self.grad_ouputput_list = []

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class LSQDPlusTestModel2(nn.Module):
    def __init__(self):
        super(LSQDPlusTestModel2, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = LSQDPlusConv2d(
                3, 8, 3, 1, 1, bias= None,
                nbits_w=4,
                init_method_w = 2,
                customer_backward_w = False, 
                nbits_a=4,
                init_method_a = 2,
                customer_backward_a = False,
                add_offset = True, 
                momentum = 1.0, 
                debug = False)
        torch.manual_seed(seed)
        self.fc = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    r""""Method1"""
    torch.manual_seed(seed)
    x1 = torch.randn((10,3, 5, 5))
    x1.requires_grad_()
    y1 = torch.randn((10, 10))
    test_model1 = LSQDPlusTestModel1()
    hookB_1 = [Hook(layer[1], backward=True) for layer in list(test_model1._modules.items())]
    torch.manual_seed(seed)
    out1 = test_model1(x1)
    loss1 = torch.abs(y1 - out1).sum()
    loss1.backward()

    r""""Method2"""
    torch.manual_seed(seed)
    x2 = torch.randn((10,3, 5, 5))
    x2.requires_grad_()
    y2 = y1.clone()
    test_model2 = LSQDPlusTestModel2()
    hookB_2 = [Hook(layer[1],backward=True) for layer in list(test_model2._modules.items())]
    torch.manual_seed(seed)
    out2 = test_model2(x2)
    loss2 = torch.abs(y2 - out2 ).sum()
    loss2.backward()
    
    r""""Backward compare"""
    for idx in range(len(hookB_1)):
        for idx2, v in enumerate(hookB_1[idx].input):
            if not isinstance(hookB_1[idx].input[idx2], torch.Tensor):
                continue
            if not torch.equal(hookB_1[idx].input[idx2], hookB_2[idx].input[idx2]):
                print("input Not same", hookB_1[idx].name)
        
        for idx2, v in enumerate(hookB_1[idx].output):
            if not isinstance(hookB_1[idx].input[idx2], torch.Tensor):
                continue
            if not torch.equal(hookB_1[idx].output[idx2], hookB_2[idx].output[idx2]):
                print("output NOT same", hookB_1[idx].name)

    print(torch.equal(test_model1.conv1.weight.grad, test_model2.conv1.weight.grad))
    assert(loss1 == loss2)
    assert(torch.equal(test_model1.fc.weight, test_model2.fc.weight))
    assert(torch.equal(test_model1.fc.weight.grad, test_model2.fc.weight.grad))

    assert(torch.equal(test_model1.conv1.weight, test_model2.conv1.weight))
    assert(torch.equal(test_model1.conv1.weight_alpha, test_model2.conv1.weight_alpha)) 
    assert(torch.equal(test_model1.conv1.weight_offset, test_model2.conv1.weight_offset)) 
    assert(torch.equal(test_model1.conv1.act_alpha, test_model2.conv1.act_alpha))  
    assert(torch.equal(test_model1.conv1.act_offset, test_model2.conv1.act_offset)) 

    temp1 = test_model2.conv1.weight.grad.flatten() == test_model1.conv1.weight.grad.flatten()
    temp1_dix = torch.where(temp1 == False)
    diff_weight_value1 = test_model2.conv1.weight.flatten()[temp1 == False].detach().numpy()
    for idx, val in enumerate(test_model1.conv1.weight.grad.flatten()):
        if val != test_model2.conv1.weight.grad.flatten()[idx]:
            break

    print(torch.equal(test_model1.conv1.act_alpha.grad, test_model2.conv1.act_alpha.grad))