import torch
import torch.nn as nn
from ..builder import QUANLAYERS
import torch.nn.functional as F


@QUANLAYERS.register_module()
class DRFLinear(nn.Linear):
    """Applies a quantized linear layers in DoReFa-Net way.

    Args:
        in_features (int):  Size of each input sample
        out_features (int): Size of each output sample
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        bit_w(int, optional): Bitwidth for the weight quantization. Default: ``8``
        bit_a(int, optional): Bitwidth for the activation quantization. Default: ``8``
        quant_activation(bool, optional): If ``True``, quantizes activation. Default: ``True``

    Procedure:
       
        Weight quantization
        .. math::
            w_m = \frac{\tanh{weight}}{2 \max{\| \tanh{weight} \|}},
            output = \min{quant_max, \max{quant_min, round(w_m / scale)}}.
            
        Activation quantization
        .. math::
            a_m = clip(activation, 0, 1),
            output = \min{quant_max, \max{quant_min, round(a_m / scale)}}.

        `quant_max`:  upper bound of the quantized domain
        `quant_min`:  lower bound of the quantized domain
        `scale`: quantization scale

    """

    def __init__(self, in_features, out_features, bias=True, nbits_w=8, nbits_a=8):
        super(DRFLinear, self).__init__(in_features, out_features, bias=bias)
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
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
        x_q = self.drf_quantization_act(x)    
        return F.linear(x_q, w_q, self.bias)
