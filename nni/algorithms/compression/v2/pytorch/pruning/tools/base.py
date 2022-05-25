# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from datetime import datetime
import logging
from pathlib import Path
import types
from typing import List, Dict, Tuple, Optional, Callable, Union

import json_tricks
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from ...base import Pruner, LayerInfo, Task, TaskResult
from ...utils import OptimizerConstructHelper, Scaling

_logger = logging.getLogger(__name__)


class DataCollector:
    """
    An abstract class for collect the data needed by the compressor.

    Parameters
    ----------
    compressor
        The compressor binded with this DataCollector.
    """

    def __init__(self, compressor: Pruner):
        self.compressor = compressor

    def reset(self):
        """
        Reset the `DataCollector`.
        """
        raise NotImplementedError()

    def collect(self) -> Dict:
        """
        Collect the compressor needed data, i.e., module weight, the output of activation function.

        Returns
        -------
        Dict
            Usually has format like {module_name: tensor_type_data}.
        """
        raise NotImplementedError()


class HookCollectorInfo:
    def __init__(self, targets: Union[Dict[str, Tensor], List[LayerInfo]], hook_type: str,
                 collector: Union[Callable[[List, Tensor], Callable[[Tensor], None]], Callable[[List], Callable[[Module, Tensor, Tensor], None]]]):
        """
        This class used to aggregate the information of what kind of hook is placed on which layers.

        Parameters
        ----------
        targets
            List of LayerInfo or Dict of {layer_name: weight_tensor}, the hook targets.
        hook_type
            'forward' or 'backward'.
        collector
            A hook function generator, the input is a buffer (empty list) or a buffer (empty list) and tensor, the output is a hook function.
            The buffer is used to store the data wanted to hook.
        """
        self.targets = targets
        self.hook_type = hook_type
        self.collector = collector


class TrainerBasedDataCollector(DataCollector):
    """
    This class includes some trainer based util functions, i.e., patch optimizer or criterion, add hooks.
    """

    def __init__(self, compressor: Pruner, trainer: Callable[[Module, Optimizer, Callable], None], optimizer_helper: OptimizerConstructHelper,
                 criterion: Callable[[Tensor, Tensor], Tensor], training_epochs: int,
                 opt_before_tasks: List = [], opt_after_tasks: List = [],
                 collector_infos: List[HookCollectorInfo] = [], criterion_patch: Optional[Callable[[Callable], Callable]] = None):
        """
        Parameters
        ----------
        compressor
            The compressor binded with this DataCollector.
        trainer
            A callable function used to train model or just inference. Take model, optimizer, criterion as input.
            The model will be trained or inferenced `training_epochs` epochs.

            Example::

                def trainer(model: Module, optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]):
                    training = model.training
                    model.train(mode=True)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        # If you don't want to update the model, you can skip `optimizer.step()`, and set train mode False.
                        optimizer.step()
                    model.train(mode=training)
        optimizer
            The optimizer instance used in trainer. Note that this optimizer might be patched during collect data,
            so do not use this optimizer in other places.
        criterion
            The criterion function used in trainer. Take model output and target value as input, and return the loss.
        training_epochs
            The total number of calling trainer.
        opt_before_tasks
            A list of function that will be called one by one before origin `optimizer.step()`.
            Note that these functions will be patched into `optimizer.step()`.
        opt_after_tasks
            A list of function that will be called one by one after origin `optimizer.step()`.
            Note that these functions will be patched into `optimizer.step()`.
        collector_infos
            A list of `HookCollectorInfo` instance. And the hooks will be registered in `__init__`.
        criterion_patch
            A callable function used to patch the criterion. Take a criterion function as input and return a new one.

            Example::

                def criterion_patch(criterion: Callable[[Tensor, Tensor], Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
                    weight = ...
                    def patched_criterion(output, target):
                        return criterion(output, target) + torch.norm(weight)
                    return patched_criterion
        """
        super().__init__(compressor)
        self.trainer = trainer
        self.training_epochs = training_epochs
        self.optimizer_helper = optimizer_helper
        self._origin_criterion = criterion
        self._opt_before_tasks = opt_before_tasks
        self._opt_after_tasks = opt_after_tasks

        self._criterion_patch = criterion_patch

        self.reset(collector_infos)

    def reset(self, collector_infos: List[HookCollectorInfo] = []):
        # refresh optimizer and criterion
        self._reset_optimizer()

        if self._criterion_patch is not None:
            self.criterion = self._criterion_patch(self._origin_criterion)
        else:
            self.criterion = self._origin_criterion

        # patch optimizer
        self._patch_optimizer()

        # hook
        self._remove_all_hook()
        self._hook_id = 0
        self._hook_handles = {}
        self._hook_buffer = {}

        self._collector_infos = collector_infos
        self._add_all_hook()

    def _reset_optimizer(self):
        parameter_name_map = self.compressor.get_origin2wrapped_parameter_name_map()
        assert self.compressor.bound_model is not None
        self.optimizer = self.optimizer_helper.call(self.compressor.bound_model, parameter_name_map)

    def _patch_optimizer(self):
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
        if self.optimizer is not None:
            self.optimizer.step = types.MethodType(patch_step(self.optimizer.step), self.optimizer)

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


class MetricsCalculator:
    """
    An abstract class for calculate a kind of metrics of the given data.

    Parameters
    ----------
    dim
        The dimensions that corresponding to the under pruning weight dimensions in collected data.
        None means one-to-one correspondence between pruned dimensions and data, which equal to set `dim` as all data dimensions.
        Only these `dim` will be kept and other dimensions of the data will be reduced.

        Example:

        If you want to prune the Conv2d weight in filter level, and the weight size is (32, 16, 3, 3) [out-channel, in-channel, kernal-size-1, kernal-size-2].
        Then the under pruning dimensions is [0], which means you want to prune the filter or out-channel.

            Case 1: Directly collect the conv module weight as data to calculate the metric.
            Then the data has size (32, 16, 3, 3).
            Mention that the dimension 0 of the data is corresponding to the under pruning weight dimension 0.
            So in this case, `dim=0` will set in `__init__`.

            Case 2: Use the output of the conv module as data to calculate the metric.
            Then the data has size (batch_num, 32, feature_map_size_1, feature_map_size_2).
            Mention that the dimension 1 of the data is corresponding to the under pruning weight dimension 0.
            So in this case, `dim=1` will set in `__init__`.

        In both of these two case, the metric of this module has size (32,).

    block_sparse_size
        This used to describe the block size a metric value represented. By default, None means the block size is ones(len(dim)).
        Make sure len(dim) == len(block_sparse_size), and the block_sparse_size dimension position is corresponding to dim.

        Example:

        The under pruning weight size is (768, 768), and you want to apply a block sparse on dim=[0] with block size [64, 768],
        then you can set block_sparse_size=[64]. The final metric size is (12,).
    """

    def __init__(self, dim: Optional[Union[int, List[int]]] = None,
                 block_sparse_size: Optional[Union[int, List[int]]] = None):
        self.dim = dim if not isinstance(dim, int) else [dim]
        self.block_sparse_size = block_sparse_size if not isinstance(block_sparse_size, int) else [block_sparse_size]
        if self.block_sparse_size is not None:
            assert all(i >= 1 for i in self.block_sparse_size)
        elif self.dim is not None:
            self.block_sparse_size = [1] * len(self.dim)
        if self.dim is not None:
            assert all(i >= 0 for i in self.dim)
            self.dim, self.block_sparse_size = (list(t) for t in zip(*sorted(zip(self.dim, self.block_sparse_size))))  # type: ignore

    def calculate_metrics(self, data: Dict) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        data
            A dict handle the data used to calculate metrics. Usually has format like {module_name: tensor_type_data}.

        Returns
        -------
        Dict[str, Tensor]
            The key is the layer_name, value is the metric.
            Note that the metric has the same size with the data size on `dim`.
        """
        raise NotImplementedError()


class SparsityAllocator:
    """
    A base class for allocating mask based on metrics.

    Parameters
    ----------
    pruner
        The pruner that binded with this `SparsityAllocator`.
    scalors
        Scalor is used to scale the mask. It shrinks the mask of the same size as the pruning target to the same size as the metric,
        or expands the mask of the same size as the metric to the same size as the pruning target.
        If you want to use different scalors for different pruning targets, please use a dict `{target_name: scalor}`.
        If allocator meets an unspecified target, it will use `scalors['_default']` to scale its mask.
        Passing in scalor instead of a `dict` will be considered passed in `{'_default': scalor}`.
        Passing in `None` means no need to scale.
    continuous_mask
        If set True, the part that has been masked will be masked first.
        If set False, the part that has been masked may be unmasked due to the increase of its corresponding metric.
    """

    def __init__(self, pruner: Pruner, scalors: Dict[str, Scaling] | Scaling | None = None, continuous_mask: bool = True):
        self.pruner = pruner
        self.scalors: Dict[str, Scaling] | None = scalors if isinstance(scalors, (dict, None)) else {'_default': scalors}
        self.continuous_mask = continuous_mask

    def _get_scalor(self, module_name: str, target_name: str) -> Scaling | None:
        # Get scalor for the specific target in the specific module. Return None if don't find it.
        if self.scalors:
            return self.scalors.get(target_name, self.scalors.get('_default', None))
        else:
            return None

    def _expand_mask(self, module_name: str, target_name: str, mask: Tensor) -> Tensor:
        # Expand the shrinked mask to the pruning target size.
        scalor = self._get_scalor(module_name=module_name, target_name=target_name)
        if scalor:
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            return scalor.expand(mask, getattr(wrapper, f'{target_name}_mask').shape)
        else:
            return mask.clone()

    def _shrink_mask(self, module_name: str, target_name: str, mask: Tensor) -> Tensor:
        # Shrink the mask by scalor, shrinked mask usually has the same size with metric. 
        scalor = self._get_scalor(module_name=module_name, target_name=target_name)
        if scalor:
            mask = scalor.shrink(mask)
        return (mask != 0).type_as(mask)

    def _mask_metrics(self, metrics: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Set the already masked part in the metric to the minimum value.
        target_name = 'weight'
        for module_name, weight_metric in metrics.items():
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            old_weight_mask = getattr(wrapper, f'{target_name}_mask', None)
            if old_weight_mask:
                shrinked_mask = self._shrink_mask(module_name, target_name, old_weight_mask)
                # ensure the already masked part has the minimum value.
                min_val = weight_metric.min() - 1
                metrics[module_name] = torch.where(shrinked_mask!=0, weight_metric, min_val)
        return metrics

    def common_target_masks_generation(self, metrics: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        """
        Generate masks for metrics-dependent targets.
        """
        raise NotImplementedError()

    def special_target_masks_generation(self, masks: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        """
        Some pruning targets' mask generation depends on other targets, i.e., bias mask depends on weight mask.
        This function is used to generate these masks, and it be called at the end of `generate_sparsity`.

        Parameters
        ----------
        masks
            The format is {module_name: {target_name: mask}}
        """
        for module_name, module_masks in masks.items():
            # generate bias mask, this will move to wrapper in the future.
            weight_mask = module_masks.get('weight_mask', None)
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            old_bias_mask = getattr(wrapper, 'bias_mask', None)
            if weight_mask and old_bias_mask and weight_mask.shape[0] == old_bias_mask.shape[0]:
                # keep dim 0 and reduce all other dims by sum
                reduce_dims = [reduce_dim for reduce_dim in range(1, len(weight_mask.shape))]
                # count unmasked number of values on dim 0 (output channel) of weight
                unmasked_num_on_dim0 = weight_mask.sum(reduce_dims) if reduce_dims else weight_mask
                module_masks['bias'] = (unmasked_num_on_dim0 != 0).type_as(old_bias_mask)
        return masks

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        """
        Parameters
        ----------
        metrics
            A metric dict with format {module_name: weight_metric}

        Returns
        -------
        Dict[str, Dict[str, Tensor]]
            The masks format is {module_name: {target_name: mask}}.
        """
        if self.continuous_mask:
            metrics = self._mask_metrics(metrics)
        masks = self.common_target_masks_generation(metrics)
        masks = self.special_target_masks_generation(masks)
        return masks


class TaskGenerator:
    """
    This class used to generate config list for pruner in each iteration.

    Parameters
    ----------
    origin_model
        The origin unwrapped pytorch model to be pruned.
    origin_masks
        The pre masks on the origin model. This mask maybe user-defined or maybe generate by previous pruning.
    origin_config_list
        The origin config list provided by the user. Note that this config_list is directly config the origin model.
        This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
    log_dir
        The log directory use to saving the task generator log.
    keep_intermediate_result
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    """
    def __init__(self, origin_model: Optional[Module], origin_masks: Optional[Dict[str, Dict[str, Tensor]]] = {},
                 origin_config_list: Optional[List[Dict]] = [], log_dir: Union[str, Path] = '.', keep_intermediate_result: bool = False):
        self._log_dir = log_dir
        self._keep_intermediate_result = keep_intermediate_result

        if origin_model is not None and origin_config_list is not None and origin_masks is not None:
            self.reset(origin_model, origin_config_list, origin_masks)

    def reset(self, model: Module, config_list: List[Dict] = [], masks: Dict[str, Dict[str, Tensor]] = {}):
        assert isinstance(model, Module), 'Only support pytorch module.'

        self._log_dir_root = Path(self._log_dir, datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')).absolute()
        self._log_dir_root.mkdir(parents=True, exist_ok=True)

        self._intermediate_result_dir = Path(self._log_dir_root, 'intermediate_result')
        self._intermediate_result_dir.mkdir(parents=True, exist_ok=True)

        # save origin data in {log_dir}/origin
        self._origin_model_path = Path(self._log_dir_root, 'origin', 'model.pth')
        self._origin_masks_path = Path(self._log_dir_root, 'origin', 'masks.pth')
        self._origin_config_list_path = Path(self._log_dir_root, 'origin', 'config_list.json')
        self._save_data('origin', model, masks, config_list)

        self._task_id_candidate = 0
        self._tasks: Dict[Union[int, str], Task] = {}
        self._pending_tasks: List[Task] = self.init_pending_tasks()

        self._best_score = None
        self._best_task_id = None

        # dump self._tasks into {log_dir}/.tasks
        self._dump_tasks_info()

    def _dump_tasks_info(self):
        tasks = {task_id: task.to_dict() for task_id, task in self._tasks.items()}
        with Path(self._log_dir_root, '.tasks').open('w') as f:
            json_tricks.dump(tasks, f, indent=4)

    def _save_data(self, folder_name: str, model: Module, masks: Dict[str, Dict[str, Tensor]], config_list: List[Dict]):
        Path(self._log_dir_root, folder_name).mkdir(parents=True, exist_ok=True)
        torch.save(model, Path(self._log_dir_root, folder_name, 'model.pth'))
        torch.save(masks, Path(self._log_dir_root, folder_name, 'masks.pth'))
        with Path(self._log_dir_root, folder_name, 'config_list.json').open('w') as f:
            json_tricks.dump(config_list, f, indent=4)

    def update_best_result(self, task_result: TaskResult):
        score = task_result.score
        task_id = task_result.task_id
        task = self._tasks[task_id]
        task.score = score
        if self._best_score is None or (score is not None and score > self._best_score):
            self._best_score = score
            self._best_task_id = task_id
            with Path(task.config_list_path).open('r') as fr:
                best_config_list = json_tricks.load(fr)
            self._save_data('best_result', task_result.compact_model, task_result.compact_model_masks, best_config_list)

    def init_pending_tasks(self) -> List[Task]:
        raise NotImplementedError()

    def generate_tasks(self, task_result: TaskResult) -> List[Task]:
        raise NotImplementedError()

    def receive_task_result(self, task_result: TaskResult):
        """
        Parameters
        ----------
        task_result
            The result of the task.
        """
        task_id = task_result.task_id
        assert task_id in self._tasks, 'Task {} does not exist.'.format(task_id)
        self.update_best_result(task_result)

        self._tasks[task_id].status = 'Finished'
        self._dump_tasks_info()

        self._pending_tasks.extend(self.generate_tasks(task_result))
        self._dump_tasks_info()

        if not self._keep_intermediate_result:
            self._tasks[task_id].clean_up()

    def next(self) -> Optional[Task]:
        """
        Returns
        -------
        Optional[Task]
            Return the next task from pending tasks.
        """
        if len(self._pending_tasks) == 0:
            return None
        else:
            task = self._pending_tasks.pop(0)
            task.status = 'Running'
            self._dump_tasks_info()
            return task

    def get_best_result(self) -> Optional[Tuple[Union[int, str], Module, Dict[str, Dict[str, Tensor]], Optional[float], List[Dict]]]:
        """
        Returns
        -------
        Optional[Tuple[int, Module, Dict[str, Dict[str, Tensor]], float, List[Dict]]]
            If self._best_task_id is not None,
            return best task id, best compact model, masks on the compact model, score, config list used in this task.
        """
        if self._best_task_id is not None:
            compact_model = torch.load(Path(self._log_dir_root, 'best_result', 'model.pth'))
            compact_model_masks = torch.load(Path(self._log_dir_root, 'best_result', 'masks.pth'))
            with Path(self._log_dir_root, 'best_result', 'config_list.json').open('r') as f:
                config_list = json_tricks.load(f)
            return self._best_task_id, compact_model, compact_model_masks, self._best_score, config_list
        return None
