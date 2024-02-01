import json
import matplotlib.pyplot as plt

# Specify the path to your JSON file
file_path1  = '../mase_output/TPE/software/search_ckpts/log.json'
file_path2  = '../mase_output/BF/software/search_ckpts/log.json'

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    config_numbers = []
    accuracies = []
    for key, value in data.items():
        config_number = int(key)
        accuracy = value.get("user_attrs_software_metrics", {}).get("accuracy")
        config_numbers.append(config_number)
        accuracies.append(accuracy)
    return config_numbers, accuracies

# Load and extract data from both files
config_numbers1, accuracies1 = load_data(file_path1)
config_numbers2, accuracies2 = load_data(file_path2)

# Plotting
plt.figure(figsize=(10, 6))

# Plot data from the first file
plt.plot(config_numbers1[0:2000], accuracies1[0:2000], marker='o', color='blue', label='TPE')

# Plot data from the second file
plt.plot(config_numbers2[0:2000], accuracies2[0:2000], marker='x', color='red', label='Brute Force')

plt.title('Accuracy vs Configuration Number')
plt.xlabel('Configuration Number')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(False)
plt.show()