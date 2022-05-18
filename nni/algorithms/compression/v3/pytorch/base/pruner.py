# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict

from torch.nn import Module

from .compressor import Compressor
from .wrapper import PrunerModuleWrapper


class Pruner(Compressor):
    def _create_module_wrapper(self, module: Module, module_name: str, config: Dict[str, Any]) -> PrunerModuleWrapper:
        return PrunerModuleWrapper(module=module, module_name=module_name, config=config)
