# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Dict, List
import yaml


def _get_module_registry(reg_path: str | Path | None = None) -> Dict[str, Dict | List | str]:
    """
    Get the built-in module registry.
    TODO: merge customized registry.
    """
    reg_path = Path(Path(__file__).parent, 'module_registry.yml') if reg_path is None else reg_path
    with Path(reg_path).open('r') as fr:
        built_in_registry = yaml.safe_load(fr)
    return built_in_registry


def _inherit_config(target: Dict[str, Dict | List | str], source: Dict[str, Dict | List | str]) -> Dict[str, Dict | List | str]:
    """
    Inherit the source config to target config.
    """
    source = deepcopy(source)
    source.pop('inherit', None)
    return _nested_update(target, source)


def _nested_update(target: Dict | List, source: Dict | List) -> Dict | List:
    assert type(target) is type(source)
    if isinstance(target, dict):
        for k, v in source.items():
            target[k] = _nested_update(target[k], v) if k in target and isinstance(target[k], (list, dict)) else v
    if isinstance(target, list):
        [target.append(v) for v in source if v not in target]
    return target


def get_module_config(module_type: str, module_registry: Dict[str, Dict | List | str] | None = None,
                      reg_path: str | Path | None = None) -> Dict[str, Dict | List | str]:
    """
    Get the module config in one registry, if `inherit` is included, it will be expanded by inheritance.
    """
    module_registry = _get_module_registry(reg_path) if module_registry is None else module_registry
    module_config: Dict[str, Dict | List | str] = deepcopy(module_registry.get(module_type, module_registry['default']))
    inherit_types: List[str] = module_config.pop('inherit', [])
    for inherit_type in inherit_types:
        inherit_config = get_module_config(inherit_type, module_registry=module_registry)
        module_config = _inherit_config(module_config, inherit_config)
    return module_config


def get_full_module_target(module_type: str):
    """
    Get all targets can be compressed of one module_type.
    """
    module_config = get_module_config(module_type)
    return module_config['target']
