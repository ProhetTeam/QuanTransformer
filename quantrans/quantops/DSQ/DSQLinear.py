import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import QUANLAYERS 


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
class DSQLinear(nn.Linear):
    """Applies a quantized linear layers in DSQ way.

    args:
        in_features(int): Size of each input sample
        out_features(int): Size of each output sample
        bias(bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        momentum(float, optional): The momentum value used for the running_bound computation. Default: 0.1
        nbits_w(int): Bitwidth for the weight quantization
        nbits_a(int): Bitwidth for the activation quantization
        quant_activation(bool, optional): If ``True``, quantizes activation. Default: ``True``
        bSetQ(bool, optional): If ``True``, performs quantization. Default: ``True``
        alpha_thres(float, optinal): The threshold used for alpha clipping. Default: 0.5
        debug(bool, optional): If ``True``, saves quantized results. Default: ``False`` 
        bias_quant(bool, optional): If ``True``, quantizes bias. Default: ``False`` 

    Procedure:
       Quantization

        .. math::
            \\begin{aligned}
            \\varphi(x) & = s \\tanh{(k(x-m_i))}, \\\\
            output & = \\min{(quant_{max}, \\max{(quant_{min}, l + \\Delta (i+\\frac{\\varphi(x)+1}{2}))})}.
            \\end{aligned}
        
        `quant_max`: running upper bound of the quantized domain,

        `quant_min`: running lower bound of the quantized domain,

        :math:`\\Delta`: quantization scale.

    """
    def __init__(self, in_features, out_features, bias=True, nbits_w = 4, nbits_a = 4, QInput = True, bSetQ = True, momentum = 0.1, bias_quant = False):
        super(DSQLinear, self).__init__(in_features, out_features, bias=bias)
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.quan_input = QInput
        self.bit_range_w = 2**self.nbits_w -1	 
        self.bit_range_a = 2**self.nbits_a -1	
        self.is_quan = bSetQ        
        self.momentum = momentum
        self.bias_quant = bias_quant
        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.uW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.lW  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
            self.register_buffer('running_uw', torch.tensor([self.uW.data])) # init with uw
            self.register_buffer('running_lw', torch.tensor([self.lW.data])) # init with lw
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
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x

    def floor_pass(self, x):
        #return torch.floor(x)
        y = torch.floor(x) 
        y_grad = x
        return y.detach() - y_grad.detach() + y_grad

    def phi_function(self, x, mi, alpha, delta):

        # alpha should less than 2 or log will be None
        # alpha = alpha.clamp(None, 2)
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
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
            # Weight Part
            # moving average
            if self.training:
                cur_running_lw = self.running_lw.mul(1-self.momentum).add((self.momentum) * self.lW)
                cur_running_uw = self.running_uw.mul(1-self.momentum).add((self.momentum) * self.uW)
            else:
                cur_running_lw = self.running_lw
                cur_running_uw = self.running_uw

            Qweight = self.clipping(self.weight, cur_running_uw, cur_running_lw)
            cur_max = torch.max(Qweight)
            cur_min = torch.min(Qweight)
            delta =  (cur_max - cur_min)/(self.bit_range_w)
            #interval = (Qweight - cur_min) //delta
            interval = self.floor_pass((Qweight - cur_min) /delta)         
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.alphaW, delta)
            Qweight = self.sgn(Qweight)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval)

            Qbias = self.bias
            # Bias			
            if self.bias is not None and self.bias_quant:
                # self.running_lB.mul_(1-self.momentum).add_((self.momentum) * self.lB)
                # self.running_uB.mul_(1-self.momentum).add_((self.momentum) * self.uB)
                if self.training:
                    cur_running_lB = self.running_lB.mul(1-self.momentum).add((self.momentum) * self.lB)
                    cur_running_uB = self.running_uB.mul(1-self.momentum).add((self.momentum) * self.uB)
                else:
                    cur_running_lB = self.running_lB
                    cur_running_uB = self.running_uB

                Qbias = self.clipping(self.bias, cur_running_uB, cur_running_lB)
                cur_max = torch.max(Qbias)
                cur_min = torch.min(Qbias)
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
                                
                if self.training:                    
                    cur_running_lA = self.running_lA.mul(1-self.momentum).add((self.momentum) * self.lA)
                    cur_running_uA = self.running_uA.mul(1-self.momentum).add((self.momentum) * self.uA)
                else:
                    cur_running_lA = self.running_lA
                    cur_running_uA = self.running_uA
                    
                Qactivation = self.clipping(x, cur_running_uA, cur_running_lA)
                cur_max = torch.max(Qactivation)
                cur_min = torch.min(Qactivation)
                delta =  (cur_max - cur_min)/(self.bit_range_a)
                #interval = (Qactivation - cur_min) //delta
                interval = self.floor_pass((Qactivation - cur_min) /delta)
                mi = (interval + 0.5) * delta + cur_min                
                Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                Qactivation = self.sgn(Qactivation)
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)
                                            
            output = F.linear(Qactivation, Qweight, Qbias)
            
        else:
            output =  F.linear(x, self.weight, self.bias)

        return output
