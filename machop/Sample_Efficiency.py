import subprocess
import time

import os

def run_search_bf(config_file):
    config_path = os.path.join('configs', 'examples', config_file)
    print(config_path)
    times = []
    for _ in range(50):
        start_time = time.time()
        subprocess.run(
            f"./ch search --config configs/examples/jsc_toy_by_type_BruteForce.toml --load ../mase_output/batch_256/jsc-tiny_classification_jsc_2024-01-25/software/training_ckpts/best.ckpt --load-type pl", shell=True)
        end_time = time.time()
        times.append(end_time - start_time)

    return times

def run_search(config_file):
    config_path = os.path.join('configs', 'examples', config_file)
    print(config_path)
    times = []
    for _ in range(50):
        start_time = time.time()
        subprocess.run(
            f"./ch search --config configs/examples/jsc_toy_by_type.toml --load ../mase_output/batch_256/jsc-tiny_classification_jsc_2024-01-25/software/training_ckpts/best.ckpt --load-type pl", shell=True)
        end_time = time.time()
        times.append(end_time - start_time)

    return times


ch_path = "."  # Replace with the actual path to 'machop'

print("Running brute-force searches...")
brute_force_times = run_search_bf("jsc_toy_by_type_BruteForce.toml")

print("Running TPE-based searches...")
tpe_times = run_search("jsc_toy_by_type.toml")

average_brute_force_time = sum(brute_force_times) / len(brute_force_times)
average_tpe_time = sum(tpe_times) / len(tpe_times)

print(f"Average computation time for brute-force search: {average_brute_force_time} seconds")
print(f"Average computation time for TPE-based search: {average_tpe_time} seconds")