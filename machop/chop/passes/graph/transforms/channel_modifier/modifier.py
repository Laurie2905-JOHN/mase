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

# Not used, the method which used this caused a deepcopy error.
CHANNEL_OP = (
    "linear",
    "relu",
    "batchnorm1d",
    "batchnorm2d",
    "Conv2d",
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
    

def redefine_transform_pass(graph, pass_args=None):
    # Extract the main configuration dictionary from the passed arguments and remove it from pass_args
    main_config = pass_args.pop('config')
    # Extract and remove the default configuration; raise an error if not provided
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError("Default value must be provided.")

    # Iterate through each node in the computational graph
    for node in graph.fx_graph.nodes:
        # Retrieve the configuration for the current node, falling back to the default if the node's name is not found
        config = main_config.get(node.name, default)['config']
        # Extract the transformation name, if specified
        name = config.get("name", None)
        
        # Determine the actual target operation of the current node
        actual_target = get_node_actual_target(node)
        new_module = None  # Initialize the variable to store the new module
        
        # Process linear (fully connected) layers
        if isinstance(actual_target, nn.Linear):
            # Proceed only if a specific transformation is defined
            if name is not None:
                # Skip the node if it is named 'x' or 'output' as these are not to be transformed
                if node.target in ['x', 'output']:
                    continue
                
                # Retrieve the original module from the graph
                ori_module = graph.modules[node.target]
                # Extract the original in/out features and bias
                in_features = ori_module.in_features
                out_features = ori_module.out_features
                bias = ori_module.bias
                
                # Modify the out_features/in_features based on the specified transformation name
                match name:
                    case "output_only":
                        out_features *= config["channel_multiplier"]
                    case "both":
                        in_features *= main_config.get(config['parent'], default)['config']["channel_multiplier"]
                        out_features *= config["channel_multiplier"]
                    case "input_only":
                        in_features *= main_config.get(config['parent'], default)['config']["channel_multiplier"]
                    case _:
                        # Handle unmatched case here
                        raise ValueError(f"Unrecognized transformation name: {name}")
                
                # Instantiate a new linear module with the updated parameters
                new_module = instantiate_linear(in_features, out_features, bias)
            
        elif isinstance(actual_target, ReLU):
            name = config.get("name")
            if name:
                ori_module = graph.modules[node.target]
                new_module = instantiate_relu(ori_module.inplace)
                setattr(graph.modules[node.target], "inplace", new_module.inplace)
        
        elif isinstance(actual_target, nn.BatchNorm1d):
            name = config.get("name")
            if name:
                ori_module = graph.modules[node.target]
                # new BatchNorm1d with the original parameters
                new_module = instantiate_batchnorm(
                    ori_module.num_features, ori_module.eps, ori_module.momentum, 
                    ori_module.affine, ori_module.track_running_stats)
                parent_name, child_name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], child_name, new_module)  
                
        # Check if the current layer is a 2D batch normalization layer
        elif isinstance(actual_target, nn.BatchNorm2d):
            # Attempt to retrieve the parent configuration, if specified
            parent = config.get("parent", None)
            
            # Proceed only if a parent configuration is provided
            if parent is not None:
                # Retrieve the original batch normalization module from the graph using the node's target as the key
                ori_module = graph.modules[node.target]
        
                # Extract the original module's configuration parameters
                num_features = ori_module.num_features
                eps = ori_module.eps
                momentum = ori_module.momentum
                affine = ori_module.affine
                
                # The number of features (channels) is adjusted based on the configuration's "channel_multiplier".
                # This is crucial because:
                # - Batch normalization in 2D (nn.BatchNorm2d) operates across the channels of 2D inputs (or feature maps).
                # - Each channel or feature map has its mean and variance calculated for normalization.
                # - The "num_features" parameter must match the number of input channels to ensure each channel is normalized correctly.
                num_features *= main_config.get(parent, {}).get('config', {}).get("channel_multiplier", 1)
                
                # Create a new 2D batch normalization layer with the updated number of features and the original parameters
                new_module = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)

        elif isinstance(actual_target, nn.Conv2d):
            # name = config.get("name", None)
            if name is not None:
                ori_module = graph.modules[node.target]
                in_channels = ori_module.in_channels
                out_channels = ori_module.out_channels
                bias = ori_module.bias
                # Adjust the number of channels based on the given configuration name
                if name == "output_only":
                    # Increase the number of output channels by the specified multiplier
                    out_channels *= config["channel_multiplier"]
                elif name == "both":
                    # Increase both input and output channels by the specified multipliers
                    in_channels *= main_config.get(config['parent'], {}).get('config', {}).get("channel_multiplier", 1)
                    out_channels *= config["channel_multiplier"]
                elif name == "input_only":
                    # Increase the number of input channels by the specified multiplier
                    in_channels *= main_config.get(config['parent'], {}).get('config', {}).get("channel_multiplier", 1)
                
                # Create a new convolutional layer with the updated parameters
                new_module = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=ori_module.kernel_size, stride=ori_module.stride,
                                    padding=ori_module.padding, dilation=ori_module.dilation,
                                    groups=ori_module.groups, bias=ori_module.bias is not None,
                                    padding_mode=ori_module.padding_mode)

        if new_module is not None:
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)

    return graph, {}