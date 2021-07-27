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
class QuanTransformerV1(BaseQuanTransformer, nn.Module):
    def __init__(self, **kwargs):
        super(QuanTransformerV1, self).__init__()
        self.register_dict = OrderedDict()
        for key, value in kwargs.items():
            try:
                self.register_dict[getattr(nn, key)] = value
            except:
                print("{key} is NOT a Standard layer.")
    def set_first_layer(self):
        pass

    def set_last_layer(self):
        pass
    
    def __call__(self, model, exclude_layers =[], logger = None, **kwargs):
        r""" Convert float Model to quantization Model
        Args:
            model(nn.Module): Standard Model
            excludes_layers(list): Some layers u dnot want to quatify
            lagger: logger 
        Return:
            New Model: replace with quantization layers
        return model
        """
        if len(self.register_dict) == 0:
            logger.info(f'There is NO layer to be quantified!')
            return model

        for module_name in model._modules:
            if len(model._modules[module_name]._modules) > 0:
                self.__call__(model._modules[module_name], exclude_layers, logger, **kwargs)
            else:
                if type(getattr(model, module_name)) not in self.register_dict:
                    continue
                if module_name in exclude_layers:
                    continue
                logger.info(f"\nTransform Layer Name :{module_name} ; {type(getattr(model, module_name)).__name__} -> {self.register_dict[type(getattr(model, module_name))]['type']}")
                
                current_layer = getattr(model, module_name)
                sig = inspect.signature(type(getattr(model, module_name)))
                new_kwargs = {}
                for key in sig.parameters:
                    if sig.parameters[key].default != inspect.Parameter.empty:
                        continue
                    assert(hasattr(current_layer, key))
                    new_kwargs[key] = getattr(current_layer, key)

                new_kwargs = {**self.register_dict[type(getattr(model, module_name))], **new_kwargs} #merge two args
                new_quan_layer = build_quanlayer(new_kwargs)
                dict_merge(new_quan_layer.__dict__, current_layer.__dict__)
                setattr(model, module_name, new_quan_layer)
        return model
