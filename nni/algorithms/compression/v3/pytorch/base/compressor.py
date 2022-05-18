# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import Any, Dict, List

from torch.nn import Module

from .wrapper import ModuleWrapper
from ..config import unfold_config_list
from ..utils.common import replace_module_by_name


class Compressor:
    """
    The abstract base pytorch compressor.

    Parameters
    ----------
    model
        The model under compressed.
    config_list
        The config list used by compressor, usually specifies the 'op_types' or 'op_names' that want to compress.
    """

    def __init__(self, model: Module, config_list: List[Dict[str, Any]], *args, **kwargs):
        self.is_ready = False
        self.is_wrapped = False
        self.module_wrappers: Dict[str, ModuleWrapper] = {}
        self._config_dict: Dict[str, Dict[str, Any]] = {}

        self.bound_model = model
        self.config_list = config_list
        self._validate_config()
        self._prepare()

    def reset(self, model: Module, config_list: List[Dict[str, Any]], *args, **kwargs):
        """
        Reset the compressor with model and config_list.
        """
        if self.is_ready:
            self._clean_up()
        self.bound_model = model
        self.config_list = config_list
        self._validate_config()
        self._prepare()

    def _prepare(self):
        """
        Do all preparations before `compress`.
        """
        if not self.is_ready:
            self._config_dict = unfold_config_list(self.bound_model, self.config_list)
            # prepare all module wrappers
            for module_name, config in self._config_dict.items():
                wrapper = self._create_module_wrapper(self.bound_model, module_name, config)
                self.module_wrappers[module_name] = wrapper
                # FIXME: use logger
                print(f'[INFO] Create wrapper for {module_name}.')
            # wrap model
            self._wrap_model()
            self.is_ready = True
        else:
            # FIXME: use logger
            print(f'[Warning] {self.__class__.__name__} already prepared, no need to `_prepare`.')

    def _clean_up(self):
        """
        Clean up all the model-related attributes, keep all model-independent attributes.
        """
        if self.is_ready:
            self._unwrap_model()
            self.module_wrappers = {}
            self._config_dict = {}
            self.bound_model = None
            self.config_list = None
            self.is_ready = False
        else:
            # FIXME: use logger
            print(f'[Warning] {self.__class__.__name__} is not ready, no need to `_clean_up`.')

    def _wrap_model(self):
        """
        Wrap all modules that needed to be compressed.
        """
        if not self.is_wrapped:
            for module_name, wrapper in sorted(list(self.module_wrappers.items()), reverse=True):
                wrapper.wrap()
                replace_module_by_name(self.bound_model, module_name, wrapper)
            self.is_wrapped = True
        else:
            # FIXME: use logger
            print(f'[Warning] The bound model in {self.__class__.__name__} already wrapped, no need to `_wrap_model`.')

    def _unwrap_model(self):
        """
        Unwrap all modules that needed to be compressed.
        """
        if self.is_wrapped:
            for module_name, wrapper in sorted(list(self.module_wrappers.items())):
                wrapper.unwrap()
                replace_module_by_name(self.bound_model, module_name, wrapper.module)
            self.is_wrapped = False
        else:
            # FIXME: use logger
            print(f'[Warning] The bound model in {self.__class__.__name__} is not wrapped, no need to `_unwrap_model`.')

    def _create_module_wrapper(self, module: Module, module_name: str, config: Dict[str, Any]) -> ModuleWrapper:
        """
        Initialized the wrapper of the module.
        """
        raise NotImplementedError

    def _validate_config(self):
        """
        Optionally implement this method to check if config_list is valid.
        """
        pass

    def compress(self):
        """
        Compress the model with algorithm implemented by subclass.
        """
        raise NotImplementedError
