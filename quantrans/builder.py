'''
Author: your name
Date: 2021-07-27 15:44:34
LastEditTime: 2021-07-27 17:39:35
LastEditors: your name
Description: In User Settings Edit
FilePath: /QuantQuant/QuanTransformer/quantrans/builder.py
'''
from mmcv.utils import Registry, build_from_cfg

QUANTRANSFORMERS = Registry('quantransformer')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, dict):
        return build_from_cfg(cfg, registry, default_args)
    else:
        raise NotImplementedError

def build_mtransformer(cfg):
    return build(cfg, QUANTRANSFORMERS)

