from mmcv.utils import Registry, build_from_cfg

QUANLAYERS = Registry('quantlayer')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, dict):
        return build_from_cfg(cfg, registry, default_args)
    else:
        raise NotImplementedError

def build_quanlayer(cfg):
    return build(cfg, QUANLAYERS)
