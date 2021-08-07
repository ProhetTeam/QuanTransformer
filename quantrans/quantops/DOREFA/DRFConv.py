import torch
import torch.nn as nn
from ..builder import QUANLAYERS
import torch.nn.functional as F


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

@QUANLAYERS.register_module()
class DRFConv(nn.Conv2d):
    """Applies a quantized 2D convolution layers in DoReFa-Net way.

    args:
        in_channels(int): Number of channels in the input image, 
        out_channels(int): Number of channels produced by the convolution 
        kernel_size(int or tuple): Size of the convolving kernel 
        stride(int or tuple, optional): Stride of the convolution. Default: 1 
        padding(int, tuple or str, optional): Padding added to both sides of the input. Default: 0 
        dilation(int or tuple, optional): Spacing between kernel elements. Default: 1 
        groups(int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias(bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        nbits_w(int): Bitwidth for the weight quantization
        nbits_a(int): Bitwidth for the activation quantization
        quant_activation(bool, optional): If ``True``, quantizes activation. Default: ``True``

    procedure:
        Weight quantization

        .. math::
            \\begin{aligned}
            w_m & = \\frac{\\tanh{(weight)}}{2 \max{\| \\tanh{(weight)} \|}}, \\\\
            output & = \min{(quant_{max}, \max{(quant_{min}, round(w_m / scale))})}.
            \\end{aligned}
            
        Activation quantization

        .. math::
            \\begin{aligned}
            a_m & = clip(activation, 0, 1), \\\\
            output & = \min{(quant_{max}, \max{(quant_{min}, round(a_m / scale))})}.
            \\end{aligned}

        `quant_max`:  upper bound of the quantized domain,

        `quant_min`:  lower bound of the quantized domain,

        `scale`: quantization scale.

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
        quant_activation = True
    ):
        super(DRFConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.layer_type = 'DRFConv'
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.quant_activation = quant_activation
        self.adjust_sign = False
        self.quant_mina = - 2 ** (nbits_a - 1)
        self.quant_maxa = 2 ** (nbits_a - 1) - 1
        self.quant_minw = - 2 ** (nbits_w - 1)
        self.quant_maxw = 2 ** (nbits_w - 1) - 1
        self.max_val = 1
        self.min_val = -1 

    def _quantize(self, x, scale, quant_min, quant_max):
        x = torch.fake_quantize_per_tensor_affine(
        x, float(scale), 0, quant_min, quant_max)
        return x

    def drf_quantization_act(self, x):
        if self.nbits_a == 32:
            return x
        if torch.min(x)>=0:
            self.adjust_sign = True
            self.quant_mina = 0
            self.quant_maxa = 2 ** (self.nbits_a) - 1
        if self.nbits_a == 8:
            min_val = torch.min(x.detach())
            max_val = torch.max(x.detach())
            min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
            max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            if self.adjust_sign:
                scale = max_val_pos / (float(2 ** (self.nbits_a) - 1))
            else:
                scale = max_val_pos / (float(2 ** (self.nbits_a) - 1) / 2)
        else:
            x = torch.clamp(x, 0, 1)
            if self.adjust_sign:
                scale = torch.ones(1).to(x.device) / (float(2 ** (self.nbits_a) - 1))
            else:
                scale = torch.ones(1).to(x.device) / (float(2 ** (self.nbits_a) - 1) / 2)

        return self._quantize(x, scale, self.quant_mina, self.quant_maxa)
    
    def drf_quantization_weight(self, weight):
        if self.nbits_w == 32:
            return weight
        elif self.nbits_w == 1:
            E = torch.mean(torch.abs(weight)).detach_()
            def _sign(x, E):
                return torch.where(x==torch.tensor(0.).to(x.device), torch.ones_like(x), torch.sign(x / E)) * E
            return _sign(weight, E)
        else:
            weight = torch.tanh(weight)
            max_w = torch.max(torch.abs(weight)).detach_() + 1e-5
            weight = weight / max_w 
            min_val = torch.min(weight.detach())
            max_val = torch.max(weight.detach())
            min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
            max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(2 ** (self.nbits_w) - 1)/ 2)

        return self._quantize(weight, scale, self.quant_minw, self.quant_maxw)

    def forward(self, x):
        w_q = self.drf_quantization_weight(self.weight)
        if self.quant_activation:
            x_q = self.drf_quantization_act(x)    
        else: 
            x_q = x
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
