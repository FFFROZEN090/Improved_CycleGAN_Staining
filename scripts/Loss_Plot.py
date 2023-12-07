import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os

# Function to load log data from a file
def load_log_data(filename):
    with open(filename, 'r') as file:
        return file.read()

# Load log data from a file named log_data.txt
log_data = load_log_data('/home/frozen/Experiments_Repitition/Cell_cycleGAN/results/training_dataset_tiledGOWT_Fakulty_Inverse/cycle_gan/loss_log.txt')

# Parse log data
metrics_A = defaultdict(lambda: defaultdict(list))
metrics_B = defaultdict(lambda: defaultdict(list))

for line in log_data.strip().split('\n'):
    epoch = int(line.split(",")[0].split(":")[1])
    metrics_values = line.replace("):", ")").split(") ")[1].strip().split(" ")
    for i in range(0, len(metrics_values), 2):
        metric = metrics_values[i].replace(":", "")
        value = float(metrics_values[i + 1])
        if metric.endswith("_A"):
            metrics_A[epoch][metric].append(value)
        elif metric.endswith("_B"):
            metrics_B[epoch][metric].append(value)

# Calculate mean values
mean_metrics_A = {epoch: {metric: np.mean(values) for metric, values in metric_data.items()} for epoch, metric_data in metrics_A.items()}
mean_metrics_B = {epoch: {metric: np.mean(values) for metric, values in metric_data.items()} for epoch, metric_data in metrics_B.items()}

# Plot data_A
plt.figure(figsize=(10,5))
print(mean_metrics_A)
for metric in mean_metrics_A[1].keys():  # Assuming all epochs have the same metrics
    plt.plot(
        [epoch for epoch in mean_metrics_A.keys()],
        [mean_metrics_A[epoch][metric] for epoch in mean_metrics_A.keys()],
        label=metric
    )
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Metrics for data_A')
plt.legend()
plt.grid(True)
# Save plot to file
A_path = os.path.join('/home/frozen/Experiments_Repitition/Cell_cycleGAN/results/training_dataset_tiledGOWT_Fakulty_Inverse/cycle_gan', 'data_A.png')
plt.savefig(A_path)

# Plot data_B
plt.figure(figsize=(10,5))
for metric in mean_metrics_B[1].keys():  # Assuming all epochs have the same metrics
    plt.plot(
        [epoch for epoch in mean_metrics_B.keys()],
        [mean_metrics_B[epoch][metric] for epoch in mean_metrics_B.keys()],
        label=metric
    )
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Metrics for data_B')
plt.legend()
plt.grid(True)
# Save plot to file
B_path = os.path.join('/home/frozen/Experiments_Repitition/Cell_cycleGAN/results/training_dataset_tiledGOWT_Fakulty_Inverse/cycle_gan', 'data_B.png')
plt.savefig(B_path)