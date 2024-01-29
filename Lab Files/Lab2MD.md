## 1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr` ... You might find the doc of [torch.fx](https://pytorch.org/docs/stable/fx.html) useful.

It is used to produce a report for the graph analysis of a MaseGraph. It takes a MaseGraph as an input and counts the different node operations and module types then returns a tuple of a MaseGraph and an empty dictionary of types of operations. 

The operations are defined as follows:

1. placeholder: Represents inputs to the model; 'name' assigns input names, 'target' names the argument, 'args' nothing or default parameter of function input, 'kwargs' unused.

2. get_attr: These nodes are used to fetch parameters from your model, such as weights from layers. They locate the parameters within the model’s structure. Fetches a parameter from the module hierarchy; 'name' labels the result, 'target' identifies the parameter's location in the hierarchy, 'args' and 'kwargs' are unused.

3. call_function: These nodes represent the application of standalone functions (like torch.add) on data. They keep track of the function being used and the arguments it takes.Applies a function to values; 'name' labels the result, 'target' is the function, 'args' and 'kwargs' are the function's arguments, following Python's convention

4. call_module: These are used when a specific module (a layer in your neural network) is called. 'name' labels the result, 'target' is the module's location in the hierarchy, 'args' and 'kwargs' are arguments excluding 'self'.

5. call_method: Similar to Call_Function, but these nodes are for methods that belong to an object (like tensor.view()). They record the method being called, including the object it is called on (self) and other arguments. 'name' for labeling, 'target' is the method's name, 'args' and 'kwargs' include all method arguments including 'self'.

6. output: Correspond to the return values of functions or the final output of your model.

The function appends a network overview and layer types information to the buffer.

### 2. What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?

### profile_statistics_analysis_pass
Function performs a series of operations on a given graph section to collect profile and computes statistics (See Below) related to the weights and activations of the nodes metadata.

#### Arguments:
Graph Node Identification by Name: Targets nodes in the graph whose names match entries in target_weight_nodes or target_act_nodes for statistical analysis.

Targeting by Type or Attribute: Uses a common characteristic, defined by mase_op, to identify nodes for analysis; applicable for various operation types like convolution, pooling, etc.

target_weight_nodes: Specifies which weight layers' data should be recorded for statistical analysis.

target_act_nodes: Designates activation nodes to record statistics for.

weight_stats: Determines the type of statistics to be collected for weight nodes.

act_stats: Defines dimensions, quantile, and device for activation statistics collection.

#### Statistics:
Record: Keeps a record of all samples passed to it. It allows for samples to be moved to a specific device and adds a new dimension before concatenation if required.

VarianceOnline: Calculates the running variance and mean using Welford's online algorithm, which is more memory-efficient as it does not require storing all samples.

VariancePrecise: Computes the variance and mean by concatenating samples and using torch.var and torch.mean. It is more precise but uses more memory, which can be significant for large datasets.

RangeNSigma: Determines the range of samples within n standard deviations (sigma) from the mean. It assumes a normal distribution and can operate in either 'precise' or 'online' mode for variance calculation.

RangeMinMax: Calculates the range of samples based on the minimum and maximum values. It can also take the absolute value of samples before calculation.

RangeQuantile: Computes the range based on quantiles. It can take the absolute values of samples and reduce along specified dimensions.

AbsMean: Implements an online algorithm to compute the mean of the absolute values of the samples.

### report_node_meta_param_analysis_pass

Report Generation: Constructs a table with headers based on selected parameter categories:

Includes basic information like node name, operation type (Fx Node op), and Mase type and Mase op.

"which": Specifies which categories of parameters to include in the report (options: "all", "common", "hardware", "software").

"save_path": Defines a file path where the analysis report will be saved.

### 3. Explain why only 1 OP is changed after the `quantize_transform_pass`.

As only one call_module type (Linear) is being specified to be quantized in the transform pass, additionally, in the JSC-Tiny model, there is one linear operator. If ReLU were chosen to be quantized, then there would be 2 changes.

### 4. Write some code to traverse both `mg` and `ori_mg`, check and comment on the nodes in these two graphs. You might find the source code for the implementation of `summarize_quantization_analysis_pass` useful.

```python
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
import torch

# Iterate over pairs of nodes from the original and modified graphs
for ori_n, n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
    # Check if the node's target module has changed type after modification
    if type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n)):
        
        # Retrieve the original and quantized modules from the nodes
        ori_module = get_node_actual_target(ori_n)
        quant_module = get_node_actual_target(n)
        
        # Print the difference information
        print(f'Difference found at name: {n.name}, '
              f'MASE type: {get_mase_type(n)}, MASE operation: {get_mase_op(n)}\n'
              f'Original module: {type(ori_module)} --> '
              f'New module: {type(quant_module)}')

        # Print the weights of the original and quantized modules
        print(f'Weight of original module: {ori_module.weight}')
        print(f'Weights of quantized module: {quant_module.get_quantized_weight()}')

        # Generate a random input tensor based on the input feature size of the quantized module
        test_input = torch.randn(quant_module.in_features)
        print(f'Random generated test input: {test_input}')
        # Apply the original and quantized modules to the test input and print the outputs
        print(f'Output for original module: {ori_module(test_input)}')
        print(f'Output for quantized module: {quant_module(test_input)}')

```

### 5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the `pass_args` for your custom network might be different if you have used more than the `Linear` layer in your network.

```python
# Define batch size and model specifications
batch_size = 8
model_name = "jsc-tiny-x10"
dataset_name = "jsc"

# Initialize the data module with specified parameters
data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

# Retrieve model information and initialize the model
# Assuming get_model_info and get_model functions are pre-defined
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False
)

# Load the model from a checkpoint file
model = load_model(
    load_name="../mase_output/Lab_Output_My_Model_50_Epoch/software/training_ckpts/best-v5.ckpt",
    load_type="pl",
    model=model
)

# Initialize the input generator for the model
input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

# Generate the Mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# Run the report graph analysis pass for detailed graph information
from chop.passes.graph import report_graph_analysis_pass
_ = report_graph_analysis_pass(mg)

# Define arguments for profiling node statistics in the model
pass_args = {
    "by": "type",
    "target_weight_nodes": ["linear"],
    "target_activation_nodes": ["relu"],
    "weight_statistics": {
        "variance_precise": {"device": "cpu", "dims": "all"},
    },
    "activation_statistics": {
        "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97}
    },
    "input_generator": input_generator,
    "num_samples": 32,
}

# Perform profiling and report on node metadata and parameters
mg, _ = profile_statistics_analysis_pass(mg, pass_args)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})
```

### 6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the [Quantized Layers](../../machop/chop/passes/transforms/quantize/quantized_modules/linear.py).

```python
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
import torch

# Iterate over pairs of nodes from the original and modified graphs
for ori_n, n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
    # Check if the node's target module has changed type after modification
    if type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n)):
        
        # Retrieve the original and quantized modules from the nodes
        ori_module = get_node_actual_target(ori_n)
        quant_module = get_node_actual_target(n)
        
        # Print the difference information
        print(f'Difference found at name: {n.name}, '
              f'MASE type: {get_mase_type(n)}, MASE operation: {get_mase_op(n)}\n'
              f'Original module: {type(ori_module)} --> '
              f'New module: {type(quant_module)}')

        # Print the weights of the original and quantized modules
        print(f'Weight of original module: {ori_module.weight}')
        print(f'Weights of quantized module: {quant_module.get_quantized_weight()}')

        # Generate a random input tensor based on the input feature size of the quantized module
        test_input = torch.randn(quant_module.in_features)
        print(f'Random generated test input: {test_input}')
        # Apply the original and quantized modules to the test input and print the outputs
        print(f'Output for original module: {ori_module(test_input)}')
        print(f'Output for quantized module: {quant_module(test_input)}')

```

### 7.  Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface.

# your_dir/mase-tools/machop

# enter the following command

./ch transform \--config configs/examples/jsc_toy_by_type.toml \--task

### 8. \[Optional\] Write your own pass

Many examples of existing passes are in the [source code](../..//machop/chop/passes/__init__.py), the [test files](../../machop/test/passes) for these passes also contain useful information on helping you to understand how these passes are used.

Implement a pass to count the number of FLOPs (floating-point operations) and BitOPs (bit-wise operations).

