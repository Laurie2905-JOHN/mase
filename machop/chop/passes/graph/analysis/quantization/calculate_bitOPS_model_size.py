def calculate_model_size_and_bitops(model, lambda_w, lambda_a):
    """
    Calculates the total model size and BitOPs (Bit Operations) for the JSC_Tiny neural network model.
    This function analyzes the model's layers and computes the model size and BitOPs based on
    the layer types, their input/output features, and the fractional bit-widths for weights and activations.
    This is crucial for understanding the computational complexity and memory requirements of a quantized neural network.

    :param model: The neural network model.
    :type model: nn.Module
    :param lambda_w: Fractional bit-width for weights.
    :type lambda_w: float
    :param lambda_a: Fractional bit-width for activations.
    :type lambda_a: float

    :return: A tuple containing the total model size and BitOPs.
    :rtype: tuple
    """

    model_size = 0  # Initialize total model size (in terms of bit-width)
    bitops = 0  # Initialize total bit operations (BitOPs)

    for mg_node in graph.fx_graph.nodes:
        # Extract meta information from each node in the graph
        mase_meta_mg = mg_node.meta["mase"].parameters
        mase_op_mg = mase_meta_mg["common"]["mase_op"]
        mase_type_mg = mase_meta_mg["common"]["mase_type"]

        # if mase_type_mg in ["module", "module_related_func"]:
            

    # Check if the layer is a Linear layer since only these layers are considered for model size and BitOPs
    if isinstance(layer, nn.Linear):
        c_in = layer.in_features  # Number of input features (or channels) to the layer
        c_out = layer.out_features  # Number of output features (or channels) from the layer

        # Calculate the model size for this layer
        # Model size is computed as the product of fractional bit-width for weights and the layer's dimensions
        model_size += lambda_w * c_in * c_out

        # For BitOPs calculation, assume output feature dimensions are equal to the number of output features
        # This simplification considers the layer's operation as a 1D transformation
        o_x, o_y = c_out, 1  

        # Calculate Bit Operations for this layer
        # BitOPs are computed considering the bit-widths for weights and activations, and the layer's input/output dimensions
        bitops += lambda_w * lambda_a * c_in * c_out * o_x * o_y

    return model_size, bitops
