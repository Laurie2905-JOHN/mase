import sys
import logging 
import os
from pathlib import Path
from pprint import pprint as pp
import time

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

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


import subprocess
import psutil

def get_gpu_power_usage():
    try:
        smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits']).decode().strip()
        power_usage = [float(x) for x in smi_output.split('\n')]  # power usage in watts
        return power_usage
    except Exception as e:
        print(f"Error obtaining GPU power usage: {e}")
        return []

def get_cpu_utilization(): 
    return psutil.cpu_percent(interval=None)

# Define TDP for your CPU
cpu_tdp = 28  # in watts

recorded_accs = []
latencies = []
gpu_power_usages = []
cpu_utilizations = []  # List to store CPU utilization for each batch

cpu_tdp = 28  # in watts, your CPU's TDP
cpu_power_usages = []  # List to store estimated CPU power usage for each batch

for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    j = 0

    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs

        # Reset CPU utilization measurement
        _ = get_cpu_utilization()  # Call once to reset the measurement

        # Measure GPU power usage before prediction
        gpu_power_before = sum(get_gpu_power_usage())

        # Start measuring time
        start_time = time.time()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        preds = mg.model(xs)  # Model prediction
        
        end.record()
        torch.cuda.synchronize()  # Wait for GPU operations to finish
        end_time = time.time()
        latency = start.elapsed_time(end)  # Time in milliseconds
        latencies.append(latency)

        # Measure GPU power usage after prediction
        gpu_power_after = sum(get_gpu_power_usage())
        gpu_power_used = (gpu_power_after - gpu_power_before)
        gpu_power_usages.append(gpu_power_used)

        # Measure CPU utilization and estimate power usage
        operation_duration = end_time - start_time  # Duration of the operation
        cpu_utilization = get_cpu_utilization()  # Get CPU utilization over operation duration
        cpu_utilizations.append(cpu_utilization)
        estimated_cpu_power = (cpu_utilization / 100) * cpu_tdp * operation_duration / 3600  # Convert to kWh
        cpu_power_usages.append(estimated_cpu_power)

        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)

        if j > num_batchs:
            break
        j += 1

    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    recorded_accs.append(acc_avg)

# Analyze latencies and power usage
avg_latency = sum(latencies) / len(latencies)
avg_gpu_power_usage = sum(gpu_power_usages) / len(gpu_power_usages)
avg_cpu_utilization = sum(cpu_utilizations) / len(cpu_utilizations)
avg_cpu_power_usage = sum(cpu_power_usages) / len(cpu_power_usages)

print(f"Average Latency per Batch: {avg_latency} milliseconds")
print(f"Average GPU Power Usage per Batch: {avg_gpu_power_usage} watts")
print(f"Average CPU Utilization per Batch: {avg_cpu_utilization}%")
print(f"Average CPU Power Usage per Batch: {avg_cpu_power_usage} Wh")