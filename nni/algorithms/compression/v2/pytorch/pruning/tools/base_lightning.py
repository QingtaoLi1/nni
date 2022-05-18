from __future__ import annotations
from functools import partialmethod
import types
from typing import Callable, List

import pytorch_lightning as pl
from sqlalchemy import true
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from .base import DataCollector, HookCollectorInfo
from ...base import Pruner
from ...utils import OptimizerConstructHelper


class TrainerBasedDataCollectorL(DataCollector):
    """
    (experimental) TrainerBasedDataCollectorL is a lightning version of TrainerBasedDataCollector.
    """

    def __init__(self, compressor: Pruner, trainer: pl.Trainer, data_module: pl.LightningDataModule,
                 opt_before_tasks: List[Callable] | None = None, opt_after_tasks: List[Callable] | None = None,
                 collector_infos: List[HookCollectorInfo] | None = None, criterion_patch: Callable[[Callable], Callable] | None = None):
        """
        Parameters
        ----------
        compressor
            The compressor binded with this DataCollector.
        trainer
            A lightning trainer.
        data_module
            A lightning data module.
        opt_before_tasks
            A list of function that will be called one by one before origin `optimizer.step()`.
            Note that these functions will be patched into `optimizer.step()`.
        opt_after_tasks
            A list of function that will be called one by one after origin `optimizer.step()`.
            Note that these functions will be patched into `optimizer.step()`.
        collector_infos
            A list of `HookCollectorInfo` instance. And the hooks will be registered in `__init__`.
        criterion_patch
            A callable function used to patch the criterion. Take a function that return a loss as input and return a new one.
            Note that the original function can return a loss Tensor or a dict contained key `loss`.

            Example::

                def criterion_patch(func: Callable[[Any], Tensor | Dict]) -> Callable[[Any], Tensor | Dict]:
                    weight = ...
                    def patched_func(*args, **kwargs):
                        result = func(*args, **kwargs)
                        if isinstance(result, Tensor):
                            result += torch.norm(weight)
                        elif isinstance(result, dict):
                            result['loss'] += torch.norm(weight)
                        else:
                            raise TypeError
                        return result
                    return patched_func
        """
        super().__init__(compressor)

        self._opt_before_tasks = opt_before_tasks if opt_before_tasks is not None else []
        self._opt_after_tasks = opt_after_tasks if opt_after_tasks is not None else []
        self._criterion_patch = criterion_patch

        # lightning corresponding
        assert isinstance(self.compressor.bound_model, pl.LightningModule)
        self._trainer = trainer
        self._data_module = data_module
        self._is_patched = False

        self.reset(collector_infos)

    def reset(self, collector_infos: List[HookCollectorInfo] | None = None):
        # reset optimizer and criterion
        self._patch_configure_optimizers()
        self._patch_training_one_step()

        # hook
        self._remove_all_hook()
        self._hook_id = 0
        self._hook_handles = {}
        self._hook_buffer = {}

        self._collector_infos = collector_infos
        self._add_all_hook()

    def _patch_configure_optimizers(self):
        if not self._is_patched:
            assert isinstance(self.compressor.bound_model, pl.LightningModule)
            self._original_configure_optimizers = self.compressor.bound_model.configure_optimizers

            parameter_name_map = self.compressor.get_origin2wrapped_parameter_name_map()
            self.compressor._unwrap_model()
            # TODO: only handle one situation (return single optimizer), need to handle other five
            optimizer = self.compressor.bound_model.configure_optimizers()
            optimizer_helper: OptimizerConstructHelper = OptimizerConstructHelper.from_trace(optimizer)
            self.compressor._wrap_model()

            def new_configure_optimizers(_):
                new_optimizer = optimizer_helper.call(self.compressor.bound_model, parameter_name_map)
                self._patch_optimizer(new_optimizer)
                return new_optimizer

            self.compressor.bound_model.configure_optimizers = types.MethodType(new_configure_optimizers, self.compressor.bound_model)
            self._is_patched = True

    def _unpatch_configure_optimizers(self):
        if self._is_patched:
            assert isinstance(self.compressor.bound_model, pl.LightningModule)
            self.compressor.bound_model.configure_optimizers = self._original_configure_optimizers
            self._original_configure_optimizers = None

    def _patch_optimizer(self, optimizer: Optimizer):
        def patch_step(old_step):
            def new_step(_, *args, **kwargs):
                for task in self._opt_before_tasks:
                    task()
                # call origin optimizer step method
                output = old_step(*args, **kwargs)
                for task in self._opt_after_tasks:
                    task()
                return output
            return new_step
        if optimizer is not None:
            optimizer.step = types.MethodType(patch_step(optimizer.step), optimizer)

    def _patch_training_one_step(self):
        if self._criterion_patch is not None:
            if not self._is_patched:
                assert isinstance(self.compressor.bound_model, pl.LightningModule)

                def new_training_step(_, *args, **kwargs):
                    self._criterion_patch(self.compressor.bound_model.training_step)
                self.compressor.bound_model.training_step

    def _add_hook(self, collector_info: HookCollectorInfo) -> int:
        self._hook_id += 1
        self._hook_handles[self._hook_id] = {}
        self._hook_buffer[self._hook_id] = {}

        if collector_info.hook_type == 'forward':
            self._add_forward_hook(self._hook_id, collector_info.targets, collector_info.collector)  # type: ignore
        elif collector_info.hook_type == 'backward':
            self._add_backward_hook(self._hook_id, collector_info.targets, collector_info.collector)  # type: ignore
        elif collector_info.hook_type == 'tensor':
            self._add_tensor_hook(self._hook_id, collector_info.targets, collector_info.collector)  # type: ignore
        else:
            _logger.warning('Skip unsupported hook type: %s', collector_info.hook_type)

        return self._hook_id

    def _add_forward_hook(self, hook_id: int, layers: List[LayerInfo],
                          collector: Callable[[List], Callable[[Module, Tensor, Tensor], None]]):
        assert all(isinstance(layer_info, LayerInfo) for layer_info in layers)
        for layer in layers:
            self._hook_buffer[hook_id][layer.name] = []
            handle = layer.module.register_forward_hook(collector(self._hook_buffer[hook_id][layer.name]))
            self._hook_handles[hook_id][layer.name] = handle

    def _add_backward_hook(self, hook_id: int, layers: List[LayerInfo],
                           collector: Callable[[List], Callable[[Module, Tensor, Tensor], None]]):
        assert all(isinstance(layer_info, LayerInfo) for layer_info in layers)
        for layer in layers:
            self._hook_buffer[hook_id][layer.name] = []
            handle = layer.module.register_backward_hook(collector(self._hook_buffer[hook_id][layer.name]))  # type: ignore
            self._hook_handles[hook_id][layer.name] = handle

    def _add_tensor_hook(self, hook_id: int, tensors: Dict[str, Tensor],
                         collector: Callable[[List, Tensor], Callable[[Tensor], None]]):
        assert all(isinstance(tensor, Tensor) for _, tensor in tensors.items())
        for layer_name, tensor in tensors.items():
            self._hook_buffer[hook_id][layer_name] = []
            handle = tensor.register_hook(collector(self._hook_buffer[hook_id][layer_name], tensor))
            self._hook_handles[hook_id][layer_name] = handle

    def _remove_hook(self, hook_id: int):
        if hook_id not in self._hook_handles:
            raise ValueError("%s is not a valid collector id" % str(hook_id))
        for handle in self._hook_handles[hook_id].values():
            handle.remove()
        del self._hook_handles[hook_id]

    def _add_all_hook(self):
        for collector_info in self._collector_infos:
            self._add_hook(collector_info)

    def _remove_all_hook(self):
        if hasattr(self, '_hook_handles'):
            for hook_id in list(self._hook_handles.keys()):
                self._remove_hook(hook_id)
