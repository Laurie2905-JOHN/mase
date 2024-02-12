### Why is it unusual to sequence three linear layers consecutively without nonlinear activation functions?

When multiple linear layers are stacked together without any nonlinear activations in between, their combined effect is equivalent to that of a single linear layer. Mathematically, if you have three linear layers defined as L_1, L_2, and L_3 applying them consecutively to an input with no nonlinear activation, these layers could be represented as one L_4 layer. This does not add any additional complexity or learning capacity to the model beyond what a single linear layer could achieve.

### 1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.

nn.ReLU handles the size of the output layer automatically. Therefore, no change was required.

```python
Original Graph:
Module number 0: BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
Module number 1: ReLU(inplace=True)
Module number 2: Linear(in_features=16, out_features=16, bias=True)
Module number 3: ReLU(inplace=True)
Module number 4: Linear(in_features=16, out_features=16, bias=True)
Module number 5: ReLU(inplace=True)
Module number 6: Linear(in_features=16, out_features=5, bias=True)
Module number 7: ReLU(inplace=True)
Transformed Graph:
Module number 0: BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
Module number 1: ReLU(inplace=True)
Module number 2: Linear(in_features=16, out_features=32, bias=True)
Module number 3: ReLU(inplace=True)
Module number 4: Linear(in_features=32, out_features=32, bias=True)
Module number 5: ReLU(inplace=True)
Module number 6: Linear(in_features=32, out_features=5, bias=True)
Module number 7: ReLU(inplace=True)
```
### 2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

The code below implemented a grid search to find the best channel multiplier value. After the search space was defined the modified model was trained and results were plotted below, see Figure 1.

```python
channel_multiplier = [1, 2, 3, 4, 6, 10]
num_batchs = 1
search_spaces = []
for d_config in channel_multiplier:
    pass_config['seq_blocks_2']["config"]["channel_multiplier"] = d_config
    pass_config['seq_blocks_4']["config"]["channel_multiplier"] = d_config
    pass_config['seq_blocks_6']["config"]["channel_multiplier"] = d_config
    search_spaces.append(copy.deepcopy(pass_config))
```

The data in Figure 1 indicates that the first configuration, with a channel multiplier of 1, had the highest accuracy. Interestingly, compared to Lab3 metrics, FLOPs increase due to more neurons in the network leading to more mathematical operations. It is unclear whether training time, GPU energy, and power show reliable results because of background operations on the laptop. Due to time constraints, only one batch was completed. In future experiments, increasing the number of batches and designing a better network could yield more reliable results.

![Description of image](Metrics_Lab4.png)
*Figure 1: Metric Analysis of Different Quantization Configurations in the Model*

### 3. Can you then design a search so that it can reach a network that can have this kind of structure?

New Function:
```python
def redefine_linear_transform(graph, transform_args=None):
    config_main = transform_args
    default_config = config_main.pop('default', None)
    if default_config is None:
        raise ValueError("default configuration must be provided.")

    for index, node in enumerate(graph.fx_graph.nodes, start=1):
            node_config = config_main.get(node.name, default_config)['config']
            module_action = node_config.get("name")
            
            if module_action:
                original_module = graph.modules[node.target]
                in_features, out_features, bias = original_module.in_features, original_module.out_features, original_module.bias
                
                multiplier_in = node_config.get("channel_multiplier_in", 1)
                multiplier_out = node_config.get("channel_multiplier_out", node_config.get("channel_multiplier", 1))
                
                if module_action == "output_only":
                    out_features *= multiplier_out
                elif module_action == "both":
                    in_features *= multiplier_in
                    out_features *= multiplier_out
                elif module_action == "input_only":
                    in_features *= multiplier_in
                
                new_module = instantiate_linear(in_features, out_features, bias)
                parent_name, child_name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], child_name, new_module)
        
    return graph, {}
```
Input:

```python
transform_args = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_2": {"config": {"name": "output_only", "channel_multiplier_out": 2}},
    "seq_blocks_4": {"config": {"name": "both", "channel_multiplier_in": 2, "channel_multiplier_out": 4}},
    "seq_blocks_6": {"config": {"name": "input_only", "channel_multiplier_in": 4}},
}
```

Output:

```python
Original Graph:
Module number 0: BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
Module number 1: ReLU(inplace=True)
Module number 2: Linear(in_features=16, out_features=16, bias=True)
Module number 3: ReLU(inplace=True)
Module number 4: Linear(in_features=16, out_features=16, bias=True)
Module number 5: ReLU(inplace=True)
Module number 6: Linear(in_features=16, out_features=5, bias=True)
Module number 7: ReLU(inplace=True)
Transformed Graph:
Module number 0: BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
Module number 1: ReLU(inplace=True)
Module number 2: Linear(in_features=16, out_features=32, bias=True)
Module number 3: ReLU(inplace=True)
Module number 4: Linear(in_features=32, out_features=64, bias=True)
Module number 5: ReLU(inplace=True)
Module number 6: Linear(in_features=64, out_features=5, bias=True)
Module number 7: ReLU(inplace=True)
```

The enhanced function now facilitates non-uniform weight scaling.

### 4. Integrate the search to the chop flow, so we can run it from the command line.

The deepcopy function is designed to create a new compound object, into which it recursively inserts copies of the objects found in the original. In contrast, a shallow copy creates a new compound object but inserts references to the original objects to the extent possible. More details can be found in the Python documentation: https://docs.python.org/3/library/copy.html.

The need for deepcopy arose during a process where the graph was to be amended by the specified channel multiplier. However, using shallow copy resulted in the layers being multiplied multiple times, such as 16 x 2 in the first loop, then leading to 32 x 2 in the next etc, which would cause a dimension error.

However, using deepcopy caused an error:

RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.  If you were attempting to deepcopy a module, this may be because of a torch.nn.utils.weight_norm usage, see https://github.com/pytorch/pytorch/pull/103001

Hence, a workaround was implemented, wherein the original MaseGraph was reinstated every time the rebuild model function was called.

```python 
class ChannelSizeModifier(SearchSpaceBase):
    """
    A class to define pre-training search space for modifying the channel sizes within a model's graph.
    This is particularly useful for exploring different model architectures dynamically.
    """
    
    def _post_init_setup(self):
        """
        Setup method to initialize class variables and move the model to the GPU to avoid tensor on same device error.
        """
        self.model.to("cuda")  # Move model to CUDA
        self.mg = None  # Placeholder for the modified graph, to be generated
        self._node_info = None  # Placeholder for node information, to be populated
        self.default_config = DEFAULT_CHANNEL_MODIFIER_CONFIG  # Set the default channel modifier configuration

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        """
        Rebuilds the model according to the sampled configuration, with an option to set the model
        in evaluation mode or training mode.

        :param sampled_config: Configuration dict for modifying channel sizes.
        :param is_eval_mode: Flag to set the model in evaluation mode; defaults to True.
        :return: Modified graph with new channel sizes.
        """
        if is_eval_mode:
            self.model.eval()  # Set model to evaluation mode
        else:
            self.model.train()  # Set model to training mode

        # Self did not work, neither deepcopy so this was a workaround.
        mg = MaseGraph(self.model)
        mg, _ = init_metadata_analysis_pass(mg, None)
        mg, _ = add_common_metadata_analysis_pass(
            mg, {"dummy_in": self.dummy_input, "force_device_meta": False}
        )
        
        if sampled_config is not None:
            mg, _ = redefine_transform_pass(mg, {"config": sampled_config})  # Apply channel size modifications
        mg.model.to(self.accelerator)  # Move the modified graph's model to the specified device
        return mg

    def build_search_space(self):
        """
        Constructs the search space for channel size modification by mapping node names to their
        corresponding types and operations, and determining the choices of channel multipliers.

        :return: None. The method updates the instance's choice mappings.
        """

        # Build a mapping from node name to mase_type and mase_op.
        mase_graph = self.rebuild_model(sampled_config=None, is_eval_mode=True)

        # Build the search space
        choices = {}
        seed = self.config["seed"]

        for node in mase_graph.fx_graph.nodes:
            if node.name in seed:
                choices[node.name] = deepcopy(seed[node.name])
            else:
                choices[node.name] = deepcopy(seed["default"])

        # Flatten the choices dictionary for easier access and manipulation
        flatten_dict(choices, flattened=self.choices_flattened)
        # Calculate the length of choices for each node and store it
        self.choice_lengths_flattened = {k: len(v) for k, v in self.choices_flattened.items()}
```

Section of function definining how linear layers are ammended by the channel multiplier. The parent, defined in the toml file was a link to the previous layer so the input and output dimensions of the previous layer would link.

```python
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
                if name == "output_only":
                    out_features *= config["channel_multiplier"]
                elif name == "both":
                    in_features *= main_config.get(config['parent'], default)['config']["channel_multiplier"]
                    out_features *= config["channel_multiplier"]
                elif name == "input_only":
                    in_features *= main_config.get(config['parent'], default)['config']["channel_multiplier"]
                
                # Instantiate a new linear module with the updated parameters
                new_module = instantiate_linear(in_features, out_features, bias)
```

Below shows the output of an evaluation search on the jsc three linear layer network.

```python
Seed set to 0
+-------------------------+--------------------------+--------------------------+-----------------+--------------------------+
| Name                    |         Default          |       Config. File       | Manual Override |        Effective         |
+-------------------------+--------------------------+--------------------------+-----------------+--------------------------+
| task                    |      classification      |           cls            |                 |           cls            |
| load_name               |           None           | /home/laurie2905/mase/ma |                 | /home/laurie2905/mase/ma |
|                         |                          | se_output/Three-Linear-  |                 | se_output/Three-Linear-  |
|                         |                          | Layer/jsc-three-linear-l |                 | Layer/jsc-three-linear-l |
|                         |                          | ayers_classification_jsc |                 | ayers_classification_jsc |
|                         |                          | _2024-02-08/software/tra |                 | _2024-02-08/software/tra |
|                         |                          |  ining_ckpts/best.ckpt   |                 |  ining_ckpts/best.ckpt   |
| load_type               |            mz            |            pl            |                 |            pl            |
| batch_size              |           128            |           512            |                 |           512            |
| to_debug                |          False           |                          |                 |          False           |
| log_level               |           info           |                          |                 |           info           |
| report_to               |       tensorboard        |                          |                 |       tensorboard        |
| seed                    |            0             |            0             |                 |            0             |
| quant_config            |           None           |                          |                 |           None           |
| training_optimizer      |           adam           |                          |                 |           adam           |
| trainer_precision       |         16-mixed         |                          |                 |         16-mixed         |
| learning_rate           |          1e-05           |           0.01           |                 |           0.01           |
| weight_decay            |            0             |                          |                 |            0             |
| max_epochs              |            20            |            20            |                 |            20            |
| max_steps               |            -1            |                          |                 |            -1            |
| accumulate_grad_batches |            1             |                          |                 |            1             |
| log_every_n_steps       |            50            |            5             |                 |            5             |
| num_workers             |            8             |                          |                 |            8             |
| num_devices             |            1             |                          |                 |            1             |
| num_nodes               |            1             |                          |                 |            1             |
| accelerator             |           auto           |           gpu            |                 |           gpu            |
| strategy                |           auto           |                          |                 |           auto           |
| is_to_auto_requeue      |          False           |                          |                 |          False           |
| github_ci               |          False           |                          |                 |          False           |
| disable_dataset_cache   |          False           |                          |                 |          False           |
| target                  |   xcu250-figd2104-2L-e   |                          |                 |   xcu250-figd2104-2L-e   |
| num_targets             |           100            |                          |                 |           100            |
| is_pretrained           |          False           |                          |                 |          False           |
| max_token_len           |           512            |                          |                 |           512            |
| project_dir             | /home/laurie2905/mase/ma |                          |                 | /home/laurie2905/mase/ma |
|                         |        se_output         |                          |                 |        se_output         |
| project                 |           None           |       jsc-final-3        |                 |       jsc-final-3        |
| model                   |           None           | jsc-three-linear-layers  |                 | jsc-three-linear-layers  |
| dataset                 |           None           |           jsc            |                 |           jsc            |
+-------------------------+--------------------------+--------------------------+-----------------+--------------------------+
INFO     Initialising model 'jsc-three-linear-layers'...
INFO     Initialising dataset 'jsc'...
INFO     Project will be created at /home/laurie2905/mase/mase_output/jsc-final-3
INFO     Loaded pytorch lightning checkpoint from /home/laurie2905/mase/mase_output/Three-Linear-Layer/jsc-three-linear-layers_classification_jsc_2024-02-08/software/training_ckpts/best.ckpt
INFO     Loaded model from /home/laurie2905/mase/mase_output/Three-Linear-Layer/jsc-three-linear-layers_classification_jsc_2024-02-08/software/training_ckpts/best.ckpt.
INFO     Building search space...
INFO     Search started...
/home/laurie2905/mase/machop/chop/actions/search/strategies/optuna.py:57: ExperimentalWarning: BruteForceSampler is experimental (supported from v3.1.0). The interface can change in the future.
  sampler = optuna.samplers.BruteForceSampler()
  0%|                                                                                                                                                                | 0/5 [00:00<?, ?it/s]WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 0. Best value: 0.2:  20%|████████████████████                                                                                | 1/5 [00:00<00:03,  1.30it/s, 0.77/20000 seconds]WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 0. Best value: 0.2:  40%|████████████████████████████████████████                                                            | 2/5 [00:01<00:01,  2.06it/s, 1.06/20000 seconds]WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 0. Best value: 0.2:  60%|████████████████████████████████████████████████████████████                                        | 3/5 [00:01<00:00,  2.70it/s, 1.29/20000 seconds]WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 0. Best value: 0.2:  80%|████████████████████████████████████████████████████████████████████████████████                    | 4/5 [00:01<00:00,  2.73it/s, 1.65/20000 seconds]WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 0. Best value: 0.2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.64it/s, 1.89/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                 | hardware_metrics                                | scaled_metrics    |
|----+----------+----------------------------------+-------------------------------------------------+-------------------|
|  0 |        0 | {'loss': 1.625, 'accuracy': 0.2} | {'average_bitwidth': 32, 'memory_density': 1.0} | {'accuracy': 0.2} |
INFO     Searching is completed
(mase) (base) laurie2905@LAPTOP-LQSPNHSL:~/mase/machop$ 
```
The current implementation of MASE did not allow for training during search. The extra functionality was added so the network could be trained and then evaluated to see the modifications of channel modification on a trained network.

In the setup metric the "self.model_info.is_physical_model" was added to the _setup_metric function as well as the forward pass to enable the jsc physical model to be trained.

Need to add more here about what I changed.

```python
def _setup_metric(self):
        # Add self.model_info.is_physical_model to allow for physical models
        if self.model_info.is_vision_model or self.model_info.is_physical_model:
            match self.task:
                case "classification" | "cls":
                    self.metric = MulticlassAccuracy(
                        num_classes=self.dataset_info.num_classes
                    ).to(self.accelerator)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    self.metric = MulticlassAccuracy(
                        num_classes=self.dataset_info.num_classes
                    ).to(self.accelerator)
                case "language_modeling" | "lm":
                    self.metric = Perplexity().to(self.accelerator)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        else:
            raise ValueError(f"model type {self.model_info} is not supported.")

def forward(self, task: str, batch: dict, model):
    if self.model_info.is_vision_model or self.model_info.is_physical_model:
        match self.task:
            case "classification" | "cls":
                loss = self.vision_cls_forward(batch, model)
            case _:
                raise ValueError(f"task {self.task} is not supported.")
    elif self.model_info.is_nlp_model:
        match self.task:
            case "classification" | "cls":
                loss = self.nlp_cls_forward(batch, model)
            case "language_modeling" | "lm":
                loss = self.nlp_lm_forward(batch, model)
            case _:
                raise ValueError(f"task {self.task} is not supported.")
    else:
        raise ValueError(f"model type {self.model_info} is not supported.")

    return loss

```

Ammending the toml file to account for training and validation can be seen below this added key training information.

```python
# Section for validation
[search.strategy]
name = "optuna"
eval_mode = true
[search.strategy.sw_runner.basic_evaluation]
data_loader = "val_dataloader"
num_samples = 512

# Section for training
[search.strategy]
name = "optuna"
eval_mode = false
[search.strategy.sw_runner.basic_train]
name = "accuracy"
data_loader = "train_dataloader"
num_samples = 1000000
max_epochs = 20
lr_scheduler = "linear"
optimizer = "adam"
learning_rate = 1e-4
num_warmup_steps = 0
```

Output from training search:

```python
  sampler = optuna.samplers.BruteForceSampler()
  0%|                                                                                                                | 0/5 [00:00<?, ?it/s]WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 0. Best value: 0.480961:  20%|█████████▏                                    | 1/5 [00:26<01:45, 26.49s/it, 26.49/20000 secondsWARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 1. Best value: 0.504973:  40%|██████████████████▍                           | 2/5 [00:49<01:13, 24.51s/it, 49.62/20000 seconds]WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 1. Best value: 0.504973:  60%|███████████████████████████▌                  | 3/5 [01:15<00:50, 25.28s/it, 75.81/20000 seconds]WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 1. Best value: 0.504973:  80%|████████████████████████████████████         | 4/5 [01:40<00:24, 24.98s/it, 100.32/20000 seconds]WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 1. Best value: 0.504973: 100%|█████████████████████████████████████████████| 5/5 [02:07<00:00, 25.48s/it, 127.41/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                | scaled_metrics      |
|----+----------+------------------------------------+-------------------------------------------------+---------------------|
|  0 |        1 | {'loss': 1.332, 'accuracy': 0.505} | {'average_bitwidth': 32, 'memory_density': 1.0} | {'accuracy': 0.505} |
INFO     Searching is completed
```


### 5. Can you define a search space (maybe channel dimension) for the VGG network, and use the TPE-search to tune it?

The attempt was made, but my laptop couldn't handle the large network size. However, both the TOML configuration and the code are functional. Below is the error message I encountered:

``` python
[W 2024-02-11 22:10:21,679] Trial 0 failed with value None.                                                                                                                                
  0%|                                                                                                                                                                | 0/2 [00:57<?, ?it/s]
Traceback (most recent call last):
  File "/home/laurie2905/mase/machop/./ch", line 6, in <module>
    ChopCLI().run()
  File "/home/laurie2905/mase/machop/chop/cli.py", line 270, in run
    run_action_fn()
  File "/home/laurie2905/mase/machop/chop/cli.py", line 395, in _run_search
    search(**search_params)
  File "/home/laurie2905/mase/machop/chop/actions/search/search.py", line 92, in search
    strategy.search(search_space)
  File "/home/laurie2905/mase/machop/chop/actions/search/strategies/optuna.py", line 143, in search
    study.optimize(
  File "/home/laurie2905/anaconda3/envs/mase/lib/python3.10/site-packages/optuna/study/study.py", line 451, in optimize
    _optimize(
  File "/home/laurie2905/anaconda3/envs/mase/lib/python3.10/site-packages/optuna/study/_optimize.py", line 66, in _optimize
    _optimize_sequential(
  File "/home/laurie2905/anaconda3/envs/mase/lib/python3.10/site-packages/optuna/study/_optimize.py", line 163, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
  File "/home/laurie2905/anaconda3/envs/mase/lib/python3.10/site-packages/optuna/study/_optimize.py", line 251, in _run_trial
    raise func_err
  File "/home/laurie2905/anaconda3/envs/mase/lib/python3.10/site-packages/optuna/study/_optimize.py", line 200, in _run_trial
    value_or_values = func(trial)
  File "/home/laurie2905/mase/machop/chop/actions/search/strategies/optuna.py", line 103, in objective
    software_metrics = self.compute_software_metrics(
  File "/home/laurie2905/mase/machop/chop/actions/search/strategies/optuna.py", line 75, in compute_software_metrics
    metrics |= runner(self.data_module, model, sampled_config)
  File "/home/laurie2905/mase/machop/chop/actions/search/strategies/runners/software/eval.py", line 128, in __call__
    outputs = self.forward(batch, forward_model)
  File "/home/laurie2905/mase/machop/chop/actions/search/strategies/runners/software/eval.py", line 62, in forward
    return self.vision_cls_forward(batch, model)
  File "/home/laurie2905/mase/machop/chop/actions/search/strategies/runners/software/eval.py", line 77, in vision_cls_forward
    loss = torch.nn.functional.cross_entropy(logits, y)
  File "/home/laurie2905/anaconda3/envs/mase/lib/python3.10/site-packages/torch/nn/functional.py", line 3059, in cross_entropy
    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

