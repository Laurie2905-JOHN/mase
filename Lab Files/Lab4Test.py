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
    
    node_list = []    
    for k, node in enumerate(graph.fx_graph.nodes, 1):
        if isinstance(get_node_actual_target(node), nn.Linear):
            node_list.append(node.target)

    j = -1    
    # Iterate over each node in the graph's functional transformation (fx) representation.
    for i, node in enumerate(graph.fx_graph.nodes, 1):
        # Skip processing for nodes that are either input 'x' or 'output'.
        if node.target in ['x', 'output']:
            continue
        
        if isinstance(get_node_actual_target(node), nn.Linear):
            j += 1
        # Retrieve the configuration for the current node. Use the default configuration if specific node config is missing.
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        actual_target = get_node_actual_target(node)  # Get the actual operation type for the current node.
        
        # Check if the current node is a Linear layer and process accordingly.
        if isinstance(actual_target, nn.Linear):
            ori_module = graph.modules[node.target]  # Original module before modification.
            # Retrieve in_features and out_features from the config, falling back to original module's attributes if not specified.
            bias = ori_module.bias
            in_features_cur = config.get('in_features_cur', 16)
            out_features_cur = config.get('out_features_cur', 16)
            out_features_pre = config.get('out_features_pre', 16)
            in_features_pre = config.get('in_features_pre', 16)
            out_features_post = config.get('out_features_post', 16)
            in_features_post = config.get('in_features_post', 16)
            
            multiplier_in = config.get("channel_multiplier_in", 1)
            multiplier_out = config.get("channel_multiplier_out", 1)
            module_action = config.get("name")
            # import pdb; pdb.set_trace()
            match module_action:             
                case "output_only":
                    out_features_cur *= multiplier_out
                    in_features_post *= multiplier_out
                    new_module_cur = instantiate_linear(in_features_cur, out_features_cur, bias)
                    new_module_post = instantiate_linear(in_features_post, out_features_post, bias)
                    # import pdb; pdb.set_trace()
                    if node_list.index(node.target) + 1 == len(node_list):
                        parent_name_cur, name_cur = get_parent_name(node_list[node_list.index(node.target)])
                        setattr(graph.modules[parent_name_cur], name_cur, new_module_cur)
                    else: 
                        parent_name_cur, name_cur = get_parent_name(node_list[node_list.index(node.target)])
                        setattr(graph.modules[parent_name_cur], name_cur, new_module_cur)                             
                        parent_name_post, name_post = get_parent_name(node_list[node_list.index(node.target)+ 1])
                        setattr(graph.modules[parent_name_post], name_post, new_module_post)
                    
                    
                    # import pdb; pdb.set_trace()    
                    
                case "both":
                    in_features_pre *= multiplier_in
                    in_features_cur *= multiplier_in
                    out_features_cur *= multiplier_out
                    in_features_post *= multiplier_out
                    new_module_pre = instantiate_linear(in_features_pre, out_features_pre, bias)
                    new_module_cur = instantiate_linear(in_features_cur, out_features_cur, bias)
                    new_module_post = instantiate_linear(in_features_post, out_features_post, bias)
                    parent_name_pre, name_pre = get_parent_name(node_list[node_list.index(node.target)-1])
                    parent_name_cur, name_cur = get_parent_name(node_list[node_list.index(node.target)])
                    parent_name_post, name_post = get_parent_name(node_list[node_list.index(node.target)+1])
                    setattr(graph.modules[parent_name_pre], name_pre, new_module_pre)
                    setattr(graph.modules[parent_name_cur], name_cur, new_module_cur)
                    setattr(graph.modules[parent_name_post], name_post, new_module_post)
                    
                case "input_only":
                    out_features_pre *= multiplier_in
                    in_features_cur *= multiplier_in
                    # Create a new Linear module with the updated features and replace the original module in the graph.
                    new_module_pre = instantiate_linear(in_features_pre, out_features_pre, bias)
                    new_module_cur = instantiate_linear(in_features_cur, out_features_cur, bias)          

                    import pdb; pdb.set_trace()
                    if node_list.index(node.target) + 1 == len(node_list):
                        parent_name_cur, name_cur = get_parent_name(node_list[node_list.index(node.target)])
                        setattr(graph.modules[parent_name_cur], name_cur, new_module_cur)
                    else: 
                        parent_name_cur, name_cur = get_parent_name(node_list[node_list.index(node.target)])                             
                        parent_name_pre, name_pre = get_parent_name(node_list[node_list.index(node.target)-1])
                        setattr(graph.modules[parent_name_pre], name_pre, new_module_pre)
                        setattr(graph.modules[parent_name_cur], name_cur, new_module_cur)
                case _:
                    # Optionally handle cases that do not match any of the specified names.
                    pass