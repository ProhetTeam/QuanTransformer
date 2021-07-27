from .BaseQuanTransformer import BaseQuanTransformer
import time
import copy
import types
import inspect 
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseQuanTransformer import BaseQuanTransformer
from .builder import QUANTRANSFORMERS
from .quantops.builder import build_quanlayer
from .utils import dict_merge

@QUANTRANSFORMERS.register_module()
class QuanTransformerV2(BaseQuanTransformer, nn.Module):
    def __init__(self, 
                 quan_policy = dict(),
                 special_layers = None,
                 **kwargs):
        super(QuanTransformerV2, self).__init__()
        self.special_layers = special_layers

        self.register_dict = OrderedDict()
        for key, value in quan_policy.items():
            assert(hasattr(nn, key))
            self.register_dict[getattr(nn, key)] = value
        self.layer_idx = 0
    
    def __call__(self, model, exclude_layers =[], logger = None, prefix = 'model', **kwargs):
        r""" Convert float Model to quantization Model
        Args:
            model(nn.Module): Standard Model
            excludes_layers(list): Some layers u dnot want to quatify
            lagger: logger 
        Return:
            New Model: replace with quantization layers
        return model
        """
        if len(self.register_dict) == 0 and len(self.special_layers) == 0:
            if logger is not None:
                logger.info(f'There is NO layer to be quantified!')
            else:
                pritn('There is NO layer to be quantified!')
            return model
        
        for module_name in model._modules:
            if len(model._modules[module_name]._modules) > 0:
                self.__call__(model._modules[module_name], exclude_layers, logger, prefix + '.' + module_name,**kwargs)
            else:
                current_layer_name = (prefix + '.' + module_name)[6:]
                current_layer = getattr(model, module_name)
                if type(current_layer) not in self.register_dict and current_layer_name not in self.special_layers.layers_name:
                    continue
                
                ## 1. get parameters
                sig = inspect.signature(type(getattr(model, module_name)))
                new_kwargs = {}
                for key in sig.parameters:
                    if sig.parameters[key].default != inspect.Parameter.empty:
                        continue
                    assert(hasattr(current_layer, key))
                    new_kwargs[key] = getattr(current_layer, key)
                
                ## 2. Special layers or Normal layer
                if current_layer_name in self.special_layers.layers_name:
                    idx = self.special_layers.layers_name.index(current_layer_name)
                    quan_args = self.special_layers.convert_type[idx]
                else:
                    quan_args = self.register_dict[type(current_layer)]
                    
                new_kwargs = {**quan_args, **new_kwargs}
                new_quan_layer = build_quanlayer(new_kwargs)
                dict_merge(new_quan_layer.__dict__, current_layer.__dict__)
                setattr(model, module_name, new_quan_layer)
                self.layer_idx += 1
        return model
