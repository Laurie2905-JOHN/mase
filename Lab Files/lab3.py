import sys
import logging 
import os
from pathlib import Path
from pprint import pprint as pp
import time
# cd machop
# figure out the correct path
machop_path = Path(".").resolve().parent /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

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


logger = logging.getLogger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

pass_args = {
"by": "type",
"default": {"config": {"name": None}},
"linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},}

all_accs, all_precisions, all_recalls, all_f1s = [], [], [], []
all_losses, all_latencies, all_gpu_powers = [], [], []

import copy
# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_args))

import torch
from torchmetrics.classification import MulticlassAccuracy
from chop.passes.graph.transforms import quantize_transform_pass
import subprocess
import threading
import time
import torchmetrics
import numpy as np
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# Instantiate metrics with the specified task type
metric = MulticlassAccuracy(num_classes=5)
precision_metric = torchmetrics.Precision(num_classes=5, average='weighted', task='multiclass')
recall_metric = torchmetrics.Recall(num_classes=5, average='weighted', task='multiclass')
f1_metric = torchmetrics.F1Score(num_classes=5, average='weighted', task='multiclass')

num_batchs = 5

# Define a class for monitoring GPU power in a separate thread
class PowerMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.power_readings = []  # List to store power readings
        self.running = False      # Flag to control the monitoring loop

    def run(self):
        self.running = True
        while self.running:
            # Get current GPU power usage and append it to the list
            power = sum(get_gpu_power_usage())
            self.power_readings.append(power)
            time.sleep(0.0001)  # Wait before next reading

    def stop(self):
        self.running = False  # Stop the monitoring loop

# Create CUDA events for timing GPU operations
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Function to get current GPU power usage using NVIDIA System Management Interface
def get_gpu_power_usage():
    try:
        # Execute nvidia-smi command to get power usage
        smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits']).decode().strip()
        # Convert power usage output to a list of floats
        power_usage = [float(x) for x in smi_output.split('\n')]
        return power_usage
    except Exception as e:
        # Handle exceptions (like nvidia-smi not found)
        print(f"Error obtaining GPU power usage: {e}")
        return []

# Iterate over different configurations
for i, config in enumerate(search_spaces):
    # Apply transformations to the model based on the configuration
    mg, _ = quantize_transform_pass(mg, config)

    # Initialize lists to store metrics for each configuration
    recorded_accs = []
    latencies = []
    gpu_power_usages = []
    accs, losses = [], []
    
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
        # Record start time of the model prediction
        start.record()
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
        avg_power = sum(power_monitor.power_readings) / len(power_monitor.power_readings)

        # Store the calculated average power
        gpu_power_usages.append(avg_power)

        # Calculate accuracy and loss for the batch
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss.item())

        # Update torchmetrics metrics
        preds_labels = torch.argmax(preds, dim=1)
        precision_metric(preds_labels, ys)
        recall_metric(preds_labels, ys)
        f1_metric(preds_labels, ys)

    # Compute final precision, recall, and F1 for this configuration
    avg_precision = precision_metric.compute()
    avg_recall = recall_metric.compute()
    avg_f1 = f1_metric.compute()

    # Reset metrics for the next configuration
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    # Calculate and record average metrics for the current configuration
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    recorded_accs.append(acc_avg)
    avg_latency = sum(latencies) / len(latencies)
    avg_gpu_power_usage = sum(gpu_power_usages) / len(gpu_power_usages)

    # Print the average metrics for the current configuration
    print(f"Configuration {i}:")
    print(f"Average Accuracy: {acc_avg}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1 Score: {avg_f1}")
    print(f"Average Loss: {loss_avg}")
    print(f"Average Latency: {avg_latency} milliseconds")
    print(f"Average GPU Power Usage: {avg_gpu_power_usage} watts")

    all_accs.append(acc_avg)
    all_precisions.append(avg_precision.item())
    all_recalls.append(avg_recall.item())
    all_f1s.append(avg_f1.item())
    all_losses.append(loss_avg)
    all_latencies.append(avg_latency)
    all_gpu_powers.append(avg_gpu_power_usage)

import matplotlib.pyplot as plt

# Assuming you have a list of configurations
configurations = [f'{i}' for i in range(len(all_accs))]

# Plotting each metric in a separate line graph
plt.figure(figsize=(15, 10))

# Accuracy
plt.subplot(3, 3, 1)
plt.plot(configurations, all_accs, marker='o', color='blue', label='Accuracy')
plt.title('Accuracy per Configuration')
plt.xlabel('Configuration')
plt.ylabel('Accuracy')

# Loss
plt.subplot(3, 3, 2)
plt.plot(configurations, all_losses, marker='o', color='magenta', label='Loss')
plt.title('Loss per Configuration')
plt.xlabel('Configuration')
plt.ylabel('Loss')

# Precision
plt.subplot(3, 3, 3)
plt.plot(configurations, all_precisions, marker='o', color='red', label='Precision')
plt.title('Precision per Configuration')
plt.xlabel('Configuration')
plt.ylabel('Precision')

# Recall
plt.subplot(3, 3, 4)
plt.plot(configurations, all_recalls, marker='o', color='green', label='Recall')
plt.title('Recall per Configuration')
plt.xlabel('Configuration')
plt.ylabel('Recall')

# F1 Score
plt.subplot(3, 3, 5)
plt.plot(configurations, all_f1s, marker='o', color='cyan', label='F1 Score')
plt.title('F1 Score per Configuration')
plt.xlabel('Configuration')
plt.ylabel('F1 Score')

# Latency
plt.subplot(3, 3, 6)
plt.plot(configurations, all_latencies, marker='o', color='yellow', label='Latency')
plt.title('Latency per Configuration')
plt.xlabel('Configuration')
plt.ylabel('Latency (ms)')

# GPU Power Usage
plt.subplot(3, 3, 7)
plt.plot(configurations, all_gpu_powers, marker='o', color='orange', label='GPU Power Usage')
plt.title('GPU Power Usage per Configuration')
plt.xlabel('Configuration')
plt.ylabel('Power Usage (Watts)')

# Adjust layout for better readability
plt.tight_layout()

# Show the plot
plt.show()

