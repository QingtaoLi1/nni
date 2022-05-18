# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from ..config import get_full_module_target


class ModuleWrapper(Module):
    def __init__(self, module: Module, module_name: str, config: Dict):
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        # config information
        self.config = config

    def wrap(self):
        """
        Called by compressor, wrap the handled module.
        """
        raise NotImplementedError

    def unwrap(self):
        """
        Called by compressor, revert the handled module.
        """
        raise NotImplementedError

    def forward(self, *inputs):
        raise NotImplementedError


class PrunerModuleWrapper(ModuleWrapper):
    def __init__(self, module: Module, module_name: str, config: Dict[str, str | List | Dict]):
        super().__init__(module, module_name, config)
        self._target_names = self._get_target_names()
        self.wrap()

    def wrap(self):
        """
        Called by pruner, wrap the handled module. Parameters, buffers and forward function of the module might be modified.
        """
        self._wrap_targets()

    def unwrap(self):
        """
        Called by pruner, revert the handled module.
        """
        self._unwrap_targets()

    def _get_target_names(self):
        full_module_target = get_full_module_target(type(self.module).__name__)
        if 'prune_target' in self.config:
            recognized_names = set()
            target_names = {'parameter': [], 'buffer': [], 'special': []}
            for k, names in full_module_target.items():
                target_names[k] = [name for name in names if name in self.config['prune_target']]
                recognized_names.update(target_names[k])
            unrecognized_names = recognized_names.difference(self.config['prune_target'])
            if unrecognized_names:
                raise ValueError('Unrecognized prune targets: {}'.format(unrecognized_names))
        else:
            # by default, only prune parameter
            target_names = {'parameter': full_module_target['target']['parameter']}
        return target_names

    def _wrap_targets(self):
        for key, names in self._target_names.items():
            if key == 'parameter':
                self._wrap_parameters(names)
            else:
                raise NotImplementedError('Not supported target type {}'.format(key))

    def _wrap_parameters(self, parameters_names: List[str]):
        for name in parameters_names:
            pruning_target: Parameter | None = getattr(self, name, None)
            if pruning_target is not None:
                assert isinstance(pruning_target, Parameter), '{} is not a torch Parameter in {}.'.format(name, self.name)
                setattr(self, name, Parameter(pruning_target.data.clone()))
                delattr(self.module, name)
                self.module.register_buffer(name, pruning_target.data.clone())
                if not getattr(self, f'{name}_mask', None):
                    self.register_buffer(f'{name}_mask', torch.ones_like(pruning_target))
            else:
                self.register_buffer(f'{name}_mask', None)

    def _unwrap_targets(self):
        for key, names in self._target_names.items():
            if key == 'parameter':
                self._unwrap_parameters(names)
            else:
                raise NotImplementedError('Not supported target type {}'.format(key))

    def _unwrap_parameters(self, parameters_names: List[str]):
        for name in parameters_names:
            pruning_target: Parameter | None = getattr(self, name, None)
            if pruning_target is not None:
                assert isinstance(pruning_target, Parameter), '{} is not a torch Parameter in wrapped {}.'.format(name, self.name)
                delattr(self, name)
                delattr(self.module, name)
                setattr(self.module, name, Parameter(pruning_target.data.clone()))

    def update_masks(self, masks: Dict[str, Tensor]):
        """
        Update target masks with given masks.
        """
        for name, new_mask in masks.items():
            assert hasattr(self, f'{name}_mask')
            setattr(self, f'{name}_mask', new_mask)

    def export_masks(self) -> Dict[str, Tensor]:
        """
        Export a masks dict with format {target_name: target_mask_tensor}.
        """
        masks = {}
        for key, names in self._target_names.items():
            if key == 'parameter':
                for name in names:
                    masks[name] = getattr(self, f'{name}_mask')
            else:
                raise NotImplementedError('Not supported target type {}'.format(key))
        return masks

    def forward(self, *inputs):
        """
        The forward function running with masks. In this forward implement, the gradient will also be masked.
        """
        # mask parameters
        for key, names in self._target_names.items():
            if key == 'parameter':
                for name in names:
                    if getattr(self, name):
                        setattr(self.module, name, torch.mul(getattr(self, name), getattr(self, f'{name}_mask')))
        return self.module(*inputs)
