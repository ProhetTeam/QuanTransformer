import math
from enum import Enum
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
import random
from torch.autograd import Variable

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ApdativeQAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, lower_bound, upper_bound, coeff= 0.1):
        r"""
        LSQPlus: y = Round[clamp(x/s, n, p)] * s
        Args:
            x: input tensor
            scale: QAT scale
        return:
            x_quant: quantization tensor of x
        """
        x_hat = (x / scale).round()
        ctx.save_for_backward(x, x_hat, scale)
        ctx.constant = [lower_bound, upper_bound]
        x_hat = x_hat.clamp(lower_bound, upper_bound)
        x_quant = x_hat * scale
        quan_error = x - x_quant
        ctx.constant.append(quan_error)
        ctx.constant.append(coeff)
        return x_quant
    
    @staticmethod
    def backward(ctx, grad_output):
        r"""
        Backward:
            x_gradient: x[x > p or x < n] = 0 else 1
            scale_gradient: -(x - beta)/s + round[(x - beta)/s] else n or p
            beta_gradient: 0 or 1
        """
        x, x_hat, scale= ctx.saved_variables
        lower_bound, upper_bound, quant_error, coeff = ctx.constant

        idx_smaller = ((x / scale).round() < lower_bound)
        idx_bigger = ((x /scale).round() > upper_bound)
        g_scale = 1.0 / math.sqrt(x.numel() * upper_bound) 

        r"""1. input gradient"""
        x_grad = torch.ones_like(grad_output)# - coeff * quant_error 
        x_grad[idx_smaller] = 0
        x_grad[idx_bigger] = 0
        x_grad *= grad_output

        r"""2. scale gradient"""
        scale_grad = -x/scale + x_hat
        scale_grad[idx_smaller] = float(lower_bound)
        scale_grad[idx_bigger] = float(upper_bound)
        scale_grad = (scale_grad * grad_output).sum().unsqueeze(dim=0) * g_scale

        return x_grad, scale_grad, None, None, None

r""" 1. DEMO1 """
nbits = 4
Qn = -2 ** (nbits - 1)
Qp = 2 ** (nbits - 1) - 1

ws = []
wqs = []
alpha_grads = []
ws_grads = []
gaps = []
for i in range(-1000, 1000):
    weight = Variable(torch.Tensor([i * 0.01]), requires_grad=True)
    alpha = torch.ones(1, requires_grad=True)
    ws.append(weight.data[0])
    w_q = ApdativeQAT.apply(weight, alpha, Qn, Qp)
    wqs.append(w_q.data[0])
    w_q.backward(retain_graph=True)
    alpha_grads.append(alpha.grad.data[0])
    ws_grads.append(weight.grad.data[0])
    gaps.append((weight - w_q).data[0])

with sns.axes_style("darkgrid"):
    sns.set(rc={"figure.figsize": (16, 12)})
    df1 = pd.DataFrame(np.array([wqs, ws, gaps]).transpose(), np.array(ws), columns = ["quant-value", "float-value", "quant-error"])
    subplot(3,1,1)
    ax = sns.lineplot(data = df1)
    ax.set_ylabel("quant-value")
    ax.set_xlabel("float-value")
    
    subplot(3,1,2)
    df2 = pd.DataFrame( np.array(alpha_grads), np.array(ws), columns = ["scale-grad-value"])
    ax = sns.lineplot(data = df2)
    ax.set_ylabel("scale-grad-value")
    ax.set_xlabel("float-value")

    subplot(3,1,3)
    df3 = pd.DataFrame( np.array(ws_grads), np.array(ws), columns = ["weigths-grad-value"])
    ax = sns.lineplot(data = df3)
    ax.set_ylabel("weigths-grad-value")
    ax.set_xlabel("float-value")

    plt.savefig('test.jpg')
