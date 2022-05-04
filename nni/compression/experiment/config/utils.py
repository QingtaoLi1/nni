import json_tricks
import math
from typing import Any, Optional, Tuple, Dict, List, Type, Union

from torch.nn import Module

from nni.compression.pytorch.utils import count_flops_params
from .compression import CompressionConfig
from .vessel import CompressionVessel


KEY_MODULE_NAME = 'module_name::'
KEY_PRUNERS = 'pruners'
KEY_VESSEL = '_vessel'
KEY_ORIGINAL_TARGET = '_original_target'
KEY_THETAS = '_thetas'


def sigmoid(x: float, theta0: float = -0.5, theta1: float = 10) -> float:
    return 1 / (1 + math.exp(-theta1 * (x + theta0)))

# hard code and magic number for flops/params reward function
def _flops_theta_helper(target: Union[int, float, str, None], origin: int) -> Tuple[float, float]:
    if not target or (isinstance(target, (int, float)) and target == 0):
        return (0., 0.)
    elif isinstance(target, float):
        assert 0. < target < 1.
        return (0.1 - target, 50.)
    elif isinstance(target, int):
        assert 0 < target < origin
        return (0.1 - target / origin, 50.)
    else:
        raise NotImplementedError('Currently only supports setting the lower limit.')

# hard code and magic number for metric reward function
def _metric_theta_helper(target: Optional[float], origin: float) -> Tuple[float, float]:
    if not target:
        return (0., 0.)
    elif isinstance(target, float):
        assert 0. <= target <= 1.
        return (0.1 - target, 50.)
    else:
        raise NotImplementedError('Currently only supports setting the lower limit.')

def _summery_module_names(model: Module,
                          module_types: List[Union[Type[Module], str]],
                          module_names: List[str],
                          exclude_module_names: List[str]) -> List[str]:
    _module_types = set()
    _all_module_names = set()
    module_names_summery = set()
    if module_types:
        for module_type in module_types:
            if isinstance(module_type, Module):
                module_type = module_type.__name__
            assert isinstance(module_type, str)
            _module_types.add(module_type)

    # unfold module types as module names, add them to summery
    for module_name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in _module_types:
            module_names_summery.add(module_name)
        _all_module_names.add(module_name)

    # add module names to summery
    if module_names:
        for module_name in module_names:
            if module_name not in _all_module_names:
                # need warning, module_name not exist
                continue
            else:
                module_names_summery.add(module_name)

    # remove module names in exclude_module_names from module_names_summery
    if exclude_module_names:
        for module_name in exclude_module_names:
            if module_name not in _all_module_names:
                # need warning, module_name not exist
                continue
            if module_name in module_names_summery:
                module_names_summery.remove(module_name)

    return list(module_names_summery)

def cc_cv2ss(config: CompressionConfig, vessel: CompressionVessel):
    search_space = {}
    model, _, evaluator, dummy_input, _, _, _, _ = vessel.export()
    flops, params, results = count_flops_params(model, dummy_input, mode='full')
    metric = evaluator(model)

    module_names_summery = _summery_module_names(model, config.module_types, config.module_names, config.exclude_module_names)
    for module_name in module_names_summery:
        search_space['{}{}'.format(KEY_MODULE_NAME, module_name)] = {'_type': 'uniform', '_value': [0, 1]}

    assert not config.pruners or not config.quantizers
    # hard code for test, need remove
    search_space[KEY_PRUNERS] = {'_type': 'choice', '_value': [pruner_config.json() for pruner_config in config.pruners]}

    original_target = {'flops': flops, 'params': params, 'metric': metric, 'results': results}

    flops_theta = _flops_theta_helper(config.flops, flops)
    params_theta = _flops_theta_helper(config.params, params)
    metric_theta = _metric_theta_helper(config.metric, metric)
    thetas = {'flops': flops_theta, 'params': params_theta, 'metric': metric_theta}

    search_space[KEY_VESSEL] = {'_type': 'choice', '_value': [vessel.json()]}
    search_space[KEY_ORIGINAL_TARGET] = {'_type': 'choice', '_value': [original_target]}
    search_space[KEY_THETAS] = {'_type': 'choice', '_value': [thetas]}
    return search_space

def parse_params(kwargs: Dict[str, Any]):
    pruner_config, vessel, original_target, thetas = None, None, None, None
    config_list = []

    for key, value in kwargs.items():
        if key.startswith(KEY_MODULE_NAME):
            config_list.append({'op_names': [key.split(KEY_MODULE_NAME)[1]], 'sparsity_per_layer': float(value)})
        elif key == KEY_PRUNERS:
            pruner_config = value
        elif key == KEY_VESSEL:
            vessel = CompressionVessel(**value)
        elif key == KEY_ORIGINAL_TARGET:
            original_target = value
        elif key == KEY_THETAS:
            thetas = value

    return pruner_config, config_list, vessel, original_target, thetas

def parse_basic_pruner(pruner_config: Dict[str, str], config_list: List[Dict[str, Any]], vessel: CompressionVessel):
    model, finetuner, evaluator, dummy_input, trainer, optimizer_helper, criterion, device = vessel.export()
    if pruner_config['pruner_type'] == 'L1NormPruner':
        from nni.compression.pytorch.pruning import L1NormPruner
        basic_pruner = L1NormPruner(model=model,
                                    config_list=config_list,
                                    mode=pruner_config['mode'],
                                    dummy_input=dummy_input)
    elif pruner_config['pruner_type'] == 'TaylorFOWeightPruner':
        from nni.compression.pytorch.pruning import TaylorFOWeightPruner
        basic_pruner = TaylorFOWeightPruner(model=model,
                                            config_list=config_list,
                                            trainer=trainer,
                                            traced_optimizer=optimizer_helper,
                                            criterion=criterion,
                                            training_batches=pruner_config['training_batches'],
                                            mode=pruner_config['mode'],
                                            dummy_input=dummy_input)
    else:
        raise NotImplementedError('Unsupported basic pruner type {}'.format(pruner_config.pruner_type))
    return basic_pruner, model, finetuner, evaluator, dummy_input, device
