# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.nn import Module


WEIGHTED_MODULES = [
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'Linear', 'Bilinear',
    'PReLU',
    'Embedding', 'EmbeddingBag',
]


def get_module_by_name(model: Module, module_name: str):
    """
    Get a module specified by its module name.
    Return (parent_module, module) if the module existed, else return (None, None).
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:
            return None, None
    if hasattr(model, name_list[-1]):
        leaf_module = getattr(model, name_list[-1])
        return model, leaf_module
    else:
        return None, None


def replace_module_by_name(model: Module, module_name: str, module: Module):
    """
    Replace the module in model with a new one by the module name.
    """
    parent_module, _ = get_module_by_name(model, module_name)
    if parent_module is not None:
        name_list = module_name.split(".")
        setattr(parent_module, name_list[-1], module)
    else:
        raise Exception('{} not exist.'.format(module_name))
