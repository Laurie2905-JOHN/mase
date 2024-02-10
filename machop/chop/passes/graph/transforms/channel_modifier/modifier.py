from copy import copy, deepcopy
import logging
from torch import nn
from torch.nn import ReLU
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass

from ...utils import (
    get_node_actual_target,
    get_parent_name,
)

logger = logging.getLogger(__name__)

CHANNEL_OP = (
    "linear",
    "relu",
    "batchnorm1d",
)

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def instantiate_relu(inplace):
    return ReLU(inplace)

def instantiate_batchnorm(num_features, eps, momentum, affine, track_running_stats):
    return nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]
    

# Main function which defines transform pass on a given graph according to passed arguments.
def redefine_transform_pass(graph, pass_args=None):
    # Ensure pass_args is not None.
    if pass_args is None:
        raise ValueError("pass_args must not be None")
    
    # Extract the main configuration and the default configuration from pass_args.
    # The default configuration is used when specific node configurations are not provided.
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError("default value must be provided.")
    
    # Initialize variables to keep track of the previous input and output channel multipliers.
    pre_in, pre_out = 1, 1
    
    # Iterate over each node in the graph's functional transformation (fx) representation.
    for i, node in enumerate(graph.fx_graph.nodes, 1):
        # Skip processing for nodes that are either input 'x' or 'output'.
        if node.target in ['x', 'output']:
            continue

        # Retrieve the configuration for the current node. Use the default configuration if specific node config is missing.
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        actual_target = get_node_actual_target(node)  # Get the actual operation type for the current node.
        
        # Check if the current node is a Linear layer and process accordingly.
        if isinstance(actual_target, nn.Linear):
            ori_module = graph.modules[node.target]  # Original module before modification.
            # Retrieve in_features and out_features from the config, falling back to original module's attributes if not specified.
            in_features = config.get('in_features', ori_module.in_features)
            out_features = config.get('out_features', ori_module.out_features)
            bias = ori_module.bias  # Preserve the original bias setting.

            # Use match-case to handle different modification scenarios based on 'name'.     
            match name:
                case "output_only":
                    # Modify only the output features based on the channel multiplier.
                    in_features = ori_module.in_features
                    out_features = out_features * config["channel_multiplier"]
                    pre_out=config["channel_multiplier"] # Update the previous output multiplier for subsequent layers.
                case "both":
                    # Modify both input and output features based on the previous output and current channel multipliers.
                    in_features *= pre_out
                    out_features *= config["channel_multiplier"]
                    pre_out = pre_in
                    pre_in = config["channel_multiplier"]
                case "input_only":
                    # Modify only the input features based on the previous input channel multiplier.
                    in_features = in_features * pre_in
                    out_features = ori_module.out_features
                case _:
                    # Optionally handle cases that do not match any of the specified names.
                    pass

            # Create a new Linear module with the updated features and replace the original module in the graph.
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)

        # Process ReLU layers
        elif isinstance(actual_target, ReLU):
            name = config.get("name")
            if name:
                ori_module = graph.modules[node.target]
                new_module = instantiate_relu(ori_module.inplace)
                setattr(graph.modules[node.target], "inplace", new_module.inplace)
        
        # Process BatchNorm1d layers
        elif isinstance(actual_target, nn.BatchNorm1d):
            name = config.get("name")
            if name:
                ori_module = graph.modules[node.target]
                # Instantiate a new BatchNorm1d with the original module's parameters
                new_module = instantiate_batchnorm(
                    ori_module.num_features, ori_module.eps, ori_module.momentum, 
                    ori_module.affine, ori_module.track_running_stats)
                parent_name, child_name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], child_name, new_module)           
    return graph, {}