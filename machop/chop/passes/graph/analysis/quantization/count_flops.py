import numpy as np
import torch 
from chop.passes.graph.utils import get_node_actual_target

def calculate_flops_pass(module, in_data, out_data):
    """
    Calculates FLOPs for a given PyTorch module.
    
    Args:
        module: The PyTorch module to analyze.
        in_data: Input tensor(s) to the module.
        out_data: Output tensor(s) from the module.
        
    Returns:
        A dictionary with keys 'total_parameters', 'computations', 'backward_computations',
        'input_buffer_size', and 'output_buffer_size' detailing the computational cost of the module.
    """

    # Handle Adaptive Average Pooling 2D layer
    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        # Ensure there is exactly one input tensor
        assert len(in_data) == 1  
        # Calculate total elements in the input and output
        input_size = in_data[0].numel()  
        output_size = out_data[0].numel()  
        # Computations are equal to the number of input elements since it averages over spatial dimensions
        computations = input_size  
        backward_computations = input_size  # Backward pass computations equal to forward pass
        # Return a dictionary summarizing the computational cost
        return {
            "total_parameters": 0,  # No learnable parameters in AdaptiveAvgPool2d
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    # Handle Embedding layer
    elif isinstance(module, torch.nn.Embedding):
        # Total learnable parameters are the product of embedding dimensions and number of embeddings
        total_parameters = module.embedding_dim * module.num_embeddings  
        return {
            "total_parameters": total_parameters,
            "computations": 0,
            "backward_computations": 0,
            "input_buffer_size": 0,  
            "output_buffer_size": 0,
        }

    # Handle Average Pooling 2D and Max Pooling 2D layers
    elif isinstance(module, (torch.nn.AvgPool2d, torch.nn.MaxPool2d)):
        # Calculate the kernel window size for pooling operations
        window_size = module.kernel_size**2 if type(module.kernel_size) == int else module.kernel_size[0] * module.kernel_size[1]
        assert len(out_data) == 1  # Ensure single output tensor
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        # For  pooling layers, each output element involves computations over the window size of input elements
        computations = output_size * window_size
        backward_computations = input_size * window_size
        return {
            "total_parameters": 0,  # Pooling layers have no learnable parameters
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    # Handle Convolutional 2D layer
    elif isinstance(module, torch.nn.Conv2d):
        # Calculate the convolutional window size, taking into account the number of input channels
        _, channels, _, _ = in_data.size()
        window_size = module.kernel_size[0] * module.kernel_size[1] * channels
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        # Computations for a Conv2d layer involve multiplying the window size by the number of output elements
        computations = output_size * window_size
        backward_computations = input_size * window_size * 2  # Backward pass computations are double of forward pass
        return {
            "total_parameters": module.weight.numel(),  # Only weight parameters, bias excluded for simplicity
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

# Handle Dropout layers (both 2D and standard Dropout)
    elif isinstance(module, (torch.nn.Dropout2d, torch.nn.Dropout)):
        # Dropout does not add computational cost in terms of FLOPs as it is a 
        # simple pass-through that randomly sets input units to 0 during training.
        # No parameters to learn.
        return {
            "total_parameters": 0,
            "computations": 0,
            "backward_computations": 0,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    # Handle Linear (Fully Connected) layers
    elif isinstance(module, torch.nn.Linear):
        # The number of computations for a linear layer is determined by the matrix multiplication
        # between the input and the weight matrix, plus the addition of the bias for each output unit.
        # Compute the batch size based on input dimensions.
        batch_size = in_data[0].shape[0]
        # Computations involve multiplication and addition for each element in the weight matrix, for each batch element.
        computations = module.in_features * module.out_features * batch_size
        # If bias is used, add additional operations for the bias addition.
        if module.bias is not None:
            computations += module.out_features * batch_size
        # Backward computations are typically double the forward computations due to the gradient calculations.
        backward_computations = computations * 2
        return {
            "total_parameters": module.weight.numel() + (module.bias.numel() if module.bias is not None else 0),
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    # Handle ReLU and ReLU6 activation layers
    elif isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6)):
        # ReLU and ReLU6 activations perform a simple comparison operation for each input element.
        # No learnable parameters.
        # The computation cost is equal to the number of input elements as each element is processed individually.
        return {
            "total_parameters": 0,
            "computations": in_data[0].numel(),
            "backward_computations": in_data[0].numel(),  # Assuming the backward pass has the same computational cost.
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    # Handle Layer Normalization
    elif isinstance(module, torch.nn.LayerNorm):
        # Layer Normalization involves computation for normalization and then scaling and shifting.
        # No FLOPs for lookup operations, but the normalization process involves calculations across the normalized dimensions.
        # Assuming 5 operations per element: subtraction, division, multiplication (by gamma), addition (of beta), and the computation of mean/variance.
        computations = in_data[0].numel() * 5
        return {
            "total_parameters": module.weight.numel() + module.bias.numel(),
            "computations": computations,
            "backward_computations": computations,  # Assuming backward pass involves a similar number of operations.
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    # Handle Batch Normalization for both 1D and 2D layers
    elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        # Batch Normalization involves normalization steps similar to LayerNorm, but applied per feature map/channel.
        # Computations involve the mean and variance calculation, normalization, and then scaling and shifting.
        computations = in_data[0].numel() * 4  # Simplified to 4 operations per element
        return {
            "total_parameters": module.weight.numel() + module.bias.numel(),  # Learnable parameters are scale (gamma) and shift (beta)
            "computations": computations,
            "backward_computations": computations,  # Assuming backward pass involves a similar number of operations.
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    else:
        # Print a message for unsupported module types. Custom layers or specific types not handled here will fall into this case.
        print("Unsupported module type for analysis:", type(module))
        
def calculate_flops_mg_pass(mase_graph):
    """
    Calculates the FLOPs (Floating Point Operations) for each module in a MaseGraph.

    Args:
        mase_graph (MaseGraph): The graph representing the model for which FLOPs are calculated.

    Returns:
        tuple: The original graph and a dictionary containing the FLOPs calculation breakdown and total FLOPs.
    """
    # Dictionary to store FLOPs calculation for each module
    flops_breakdown = {}
    # Initialize total FLOPs count
    total_flops = 0
    
    # Iterate through each node in the MaseGraph
    for node in mase_graph.fx_graph.nodes:
        try:
            # Try to extract input data shape from node metadata
            input_data_shape = (node.meta['mase'].parameters['common']['args']['data_in_0']['value'],)
        except KeyError:
            # If input data shape is not found, set it as None
            input_data_shape = (None,)
        
        # Extract output data shape from node metadata
        output_data_shape = (node.meta['mase'].parameters['common']['results']['data_out_0']['value'],)

        # Get the actual PyTorch module associated with the node
        module = get_node_actual_target(node)
        
        # Check if the node is a PyTorch module to calculate FLOPs
        if isinstance(module, torch.nn.Module):
            # Calculate FLOPs for the module
            module_flops = calculate_flops_pass(module, input_data_shape, output_data_shape)
            # Store module FLOPs in the breakdown dictionary
            flops_breakdown[module] = module_flops
            # Accumulate total FLOPs
            total_flops += module_flops['computations']

    # Print FLOPs calculation breakdown and total FLOPs
    # print("FLOPs Calculation Breakdown: ", flops_breakdown)
    # print("\nTotal FLOPs: ", total_flops)

    # Return the original graph and FLOPs calculation results
    return mase_graph, {"flop_module_breakdown": flops_breakdown, "total_flops": total_flops}