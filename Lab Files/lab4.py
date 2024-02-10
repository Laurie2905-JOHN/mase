import logging
from pathlib import Path
from pprint import pprint as pp
from chop.passes.graph import report_graph_analysis_pass
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger
from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model
import logging 
import os
from pathlib import Path
from pprint import pprint as pp
import time
import pynvml
import threading
import time
import torch
from torchmetrics.classification import MulticlassAccuracy
import torchmetrics
import torch
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.utils import get_node_actual_target
from chop.passes.graph.analysis.quantization import calculate_flops_pass, calculate_flops_mg_pass

set_logging_verbosity("info")

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 512
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

from torch import nn
from chop.passes.graph.utils import get_parent_name

# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear  2
            nn.Linear(16, 16),  # linear  3
            nn.Linear(16, 5),   # linear  4
            nn.ReLU(5),  # 5
        )

    def forward(self, x):
        return self.seq_blocks(x)


model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * config["channel_multiplier"]
                out_features = out_features * config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * config["channel_multiplier"]
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}


'''


Question 1


'''

# Define the new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )
 
    def forward(self, x):
        return self.seq_blocks(x)

# Create MaseGraph from the model
model = JSC_Three_Linear_Layers()
mg = MaseGraph(model)

# Define the pass_configs
pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}

# Print the original graph
print("Original Graph:")
for block in mg.model.seq_blocks._modules:
  print(f"Module number {block}: {mg.model.seq_blocks._modules[block]}")

# Perform the transformation on the model using the pass_config dictionary
mg, _ = redefine_linear_transform_pass(
    graph=mg, pass_args={"config": pass_config})

# Print the transformed graph
print("Transformed Graph:")
for block in mg.model.seq_blocks._modules:
  print(f"Module number {block}: {mg.model.seq_blocks._modules[block]}")

'''

Question 2

'''

from chop.actions import train, test
logger = logging.getLogger(__name__)

task = "channel_multiplier"
dataset_name = "jsc"
optimizer = "adam"
max_epochs: int = 2
max_steps: int = -1
gradient_accumulation_steps: int = 1
learning_rate: float = 5e-3
weight_decay: float = 0.0
lr_scheduler_type: str = "linear"
num_warmup_steps: int = 0
save_path: str = "../mase_output/channel_mod"
auto_requeue = False
load_name: str = None
load_type: str = ""
evaluate_before_training: bool = False
visualizer = None
profile: bool = True
plt_trainer_args = {
"max_epochs": max_epochs,
"accelerator": "gpu",
}


all_accs, all_precisions, all_recalls, all_f1s = [], [], [], []
all_losses, all_latencies, all_gpu_powers, all_gpu_energy, all_flops = [], [], [], [], []


mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# Instantiate metrics with the specified task type
metric = MulticlassAccuracy(num_classes=5)
precision_metric = torchmetrics.Precision(num_classes=5, average='weighted', task='multiclass')
recall_metric = torchmetrics.Recall(num_classes=5, average='weighted', task='multiclass')
f1_metric = torchmetrics.F1Score(num_classes=5, average='weighted', task='multiclass')



# Initialize the NVIDIA Management Library (NVML)
pynvml.nvmlInit()

# Define a class for monitoring GPU power in a separate thread using pynvml
class PowerMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.power_readings = []  # List to store power readings
        self.running = False      # Flag to control the monitoring loop
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assume using GPU 0

    def run(self):
        self.running = True
        while self.running:
            # Get current GPU power usage in milliwatts and convert to watts
            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_W = power_mW / 1000.0
            self.power_readings.append(power_W)
            time.sleep(0.001)  # Wait before next reading

    def stop(self):
        self.running = False  # Stop the monitoring loop

# Create CUDA events for timing GPU operations
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}


import copy
# channel_multiplier = [2, 1, 1 ,1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
channel_multiplier = [2]
num_batchs = 1
search_spaces = []
for d_config in channel_multiplier:
    pass_config['seq_blocks_2']["config"]["channel_multiplier"] = d_config
    pass_config['seq_blocks_4']["config"]["channel_multiplier"] = d_config
    pass_config['seq_blocks_6']["config"]["channel_multiplier"] = d_config
    search_spaces.append(copy.deepcopy(pass_config))
        
eval = False
# # Number of warm-up iterations
num_warmup_iterations = 0

# Iterate over different configurations
for i, config in enumerate(search_spaces):
    
    mg = MaseGraph(model=model)
    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    mg, _ = add_software_metadata_analysis_pass(mg, None)

    # Apply transformations to the model based on the configuration
    mg, _ = redefine_linear_transform_pass(mg, pass_args={"config": config})

    # Initialize lists to store metrics for each configuration
    recorded_accs = []
    latencies = []
    gpu_power_usages = []
    accs, losses = [], []
    flops = []
    
# Iterate over batches in the training data
    for j, inputs in enumerate(data_module.train_dataloader()):

        # Break the loop after processing the specified number of batches
        if j >= num_batchs:
            break

        # Unpack inputs and labels
        xs, ys = inputs

        # Instantiate and start the power monitor
        power_monitor = PowerMonitor()
        power_monitor.start()

        torch.cuda.empty_cache()
        
        if eval:
            # Record start time of the model prediction
            start.record()
            preds = mg.model(xs)  # Run model prediction
            end.record()          # Record end time
        else:
            start.record()
            train(mg.model, model_info, data_module, data_module.dataset_info,
                task, optimizer, learning_rate, weight_decay, plt_trainer_args,
                auto_requeue, save_path, visualizer, load_name, load_type)
            preds = mg.model(xs)  # Run model prediction
            end.record()          # Record end time
            
        # Synchronize to ensure all GPU operations are finished
        torch.cuda.synchronize()

        # Calculate latency between start and end events
        latency = start.elapsed_time(end)
        latencies.append(latency)
        
        # Stop the power monitor and calculate average power
        power_monitor.stop()
        power_monitor.join()  # Ensure monitoring thread has finished
        avg_power = sum(power_monitor.power_readings) / len(power_monitor.power_readings) if power_monitor.power_readings else 0
        # Store the calculated average power
        gpu_power_usages.append(avg_power)
        
        data = calculate_flops_mg_pass(mg)
        

        # Calculate accuracy and loss for the batch
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss.item())
        flops.append(data[1]['total_flops'])
        # Update torchmetrics metrics
        preds_labels = torch.argmax(preds, dim=1)
        precision_metric(preds_labels, ys)
        recall_metric(preds_labels, ys)
        f1_metric(preds_labels, ys)

    # Compute final precision, recall, and F1 for this configuration
    avg_precision = precision_metric.compute()
    avg_recall = recall_metric.compute()
    avg_f1 = f1_metric.compute()
    
    # Compute to get correct dimensions
    avg_flops = sum(flops) / len(flops)
    
    # Reset metrics for the next configuration
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    
    if i < num_warmup_iterations:
        continue
    else:
            # Calculate and record average metrics for the current configuration
        acc_avg = sum(accs) / len(accs)
        loss_avg = sum(losses) / len(losses)
        recorded_accs.append(acc_avg)
        avg_latency = sum(latencies) / len(latencies)
        avg_gpu_power_usage = sum(gpu_power_usages) / len(gpu_power_usages)
        avg_gpu_energy_usage = (avg_gpu_power_usage / 1000) * avg_latency / (1000*3600)
        
        # Print the average metrics for the current configuration
        print(f"Configuration {i-num_warmup_iterations}:")
        print(f"Average Accuracy: {acc_avg}")
        print(f"Average Precision: {avg_precision}")
        print(f"Average Recall: {avg_recall}")
        print(f"Average F1 Score: {avg_f1}")
        print(f"Average Loss: {loss_avg}")
        if eval:
            print(f"Average Latency: {avg_latency} milliseconds")
        else:
            print(f"Average Training Time: {avg_latency/(1000*60)} minutes")

        print(f"Average GPU Power Usage: {avg_gpu_power_usage} watts")
        print(f"Average GPU Energy Usage: {avg_gpu_energy_usage} kW/hr")
        print(f"FLOPs: {avg_flops}")
        
        all_accs.append(acc_avg)
        all_precisions.append(avg_precision.item())
        all_recalls.append(avg_recall.item())
        all_f1s.append(avg_f1.item())
        all_losses.append(loss_avg)
        all_latencies.append(avg_latency)
        all_gpu_powers.append(avg_gpu_power_usage)
        all_gpu_energy.append(avg_gpu_energy_usage)
        all_flops.append(avg_flops)

import matplotlib.pyplot as plt

# Assuming list of configurations
configurations = [f'{i}' for i in range(len(all_accs))]

# Plotting each metric in a separate line graph
plt.figure(figsize=(15, 10))

# Accuracy
plt.subplot(3, 3, 1)
plt.plot(configurations, all_accs, marker='o', color='black', label='Accuracy')
plt.title('Accuracy')
plt.xlabel('Configuration')
plt.ylabel('Accuracy')

# Loss
plt.subplot(3, 3, 2)
plt.plot(configurations, all_losses, marker='o', color='black', label='Loss')
plt.title('Loss')
plt.xlabel('Configuration')
plt.ylabel('Loss')

# Precision
plt.subplot(3, 3, 3)
plt.plot(configurations, all_precisions, marker='o', color='black', label='Precision')
plt.title('Precision')
plt.xlabel('Configuration')
plt.ylabel('Precision')

# Recall
plt.subplot(3, 3, 4)
plt.plot(configurations, all_recalls, marker='o', color='black', label='Recall')
plt.title('Recall')
plt.xlabel('Configuration')
plt.ylabel('Recall')

# F1 Score
plt.subplot(3, 3, 5)
plt.plot(configurations, all_f1s, marker='o', color='black', label='F1 Score')
plt.title('F1 Score')
plt.xlabel('Configuration')
plt.ylabel('F1 Score')

# Latency
plt.subplot(3, 3, 6)
plt.plot(configurations, all_latencies, marker='o', color='black', label='Latency')
if eval:
    plt.title('Latency')
    plt.ylabel('Latency (ms)')
else:
    plt.title('Training Time')
    plt.ylabel('Training Time (ms)')
plt.xlabel('Configuration')


# GPU Power Usage
plt.subplot(3, 3, 7)
plt.plot(configurations, all_gpu_powers, marker='o', color='black', label='GPU Power Usage')
plt.title('GPU Power Usage')
plt.xlabel('Configuration')
plt.ylabel('Power Usage (Watts)')

# GPU Energy Usage
plt.subplot(3, 3, 8)
plt.plot(configurations, all_gpu_energy, marker='o', color='black', label='GPU Power Usage')
plt.title('GPU Energy Usage')
plt.xlabel('Configuration')
plt.ylabel('Energy Usage (KW/hr)')

# FLOPs
plt.subplot(3, 3, 9)
plt.plot(configurations, all_flops, marker='o', color='black', label='FLOPs')
plt.title('Total FLOPs')
plt.xlabel('Configuration')
plt.ylabel('Number of FLOPs')

# Adjust layout for better readability
plt.tight_layout()

# Show the plot
plt.show()

'''

Question 3


'''

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

# Example usage of the improved function with a simplified configuration
transform_args = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_2": {"config": {"name": "output_only", "channel_multiplier_out": 2}},
    "seq_blocks_4": {"config": {"name": "both", "channel_multiplier_in": 2, "channel_multiplier_out": 4}},
    "seq_blocks_6": {"config": {"name": "input_only", "channel_multiplier_in": 4}},
}

model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)

# Print the original graph
print("Original Graph:")
for block in mg.model.seq_blocks._modules:
  print(f"Module number {block}: {mg.model.seq_blocks._modules[block]}")

# Perform the transformation on the model using the pass_config dictionary
redefine_linear_transform(mg, transform_args=transform_args)

# Print the transformed graph
print("Transformed Graph:")
for block in mg.model.seq_blocks._modules:
  print(f"Module number {block}: {mg.model.seq_blocks._modules[block]}")