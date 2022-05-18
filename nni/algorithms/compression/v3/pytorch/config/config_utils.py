# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Any, Dict, List

from torch.nn import Module


def unfold_config_list(model: Module, config_list: List[Dict]) -> Dict[str, Dict[str, Any]]:
    '''
    Unfold config_list to op_names level, return a config_dict {op_name: config}.
    '''
    config_dict = {}
    for config in config_list:
        for key in ['op_types', 'op_names', 'exclude_op_names']:
            config.setdefault(key, [])
        op_names = []
        for module_name, module in model.named_modules():
            module_type = type(module).__name__
            if (module_type in config['op_types'] or module_name in config['op_names']) and module_name not in config['exclude_op_names']:
                op_names.append(module_name)
        config_template = deepcopy(config)
        for key in ['op_types', 'op_names', 'exclude_op_names']:
            config_template.pop(key, [])
        for op_name in op_names:
            if op_name in config_dict:
                # FIXME: use logger
                print(f'[Warning] {op_name} duplicate definition of config, replace old config:\n{config_dict[op_name]}\nwith new config:\n{config_template}\n')
            config_dict[op_name] = deepcopy(config_template)
    return config_dict
