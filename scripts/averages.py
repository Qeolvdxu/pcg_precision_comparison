import csv
import matplotlib.pyplot as plt
import numpy as np

# Create dictionaries to store sums and counts for each matrix and device
matrix_device_sums = {}
matrix_device_counts = {}

# Read the CSV file
try:
    with open('./csv_files/n4th_protection.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            matrix = row.get('MATRIX', None)
            device = row.get('DEVICE', None)
            wall_time = float(row.get('WALL_TIME', 0.0))
            mem_wall_time = float(row.get('MEM_WALL_TIME', 0.0))
            fault_time = float(row.get('FAULT_TIME', 0.0))

            if matrix is not None and device is not None:
                if device not in ['CPU', 'GPU']:
                    continue  # Skip rows with devices other than CPU and GPU

                if matrix not in matrix_device_sums:
                    matrix_device_sums[matrix] = {}
                    matrix_device_counts[matrix] = {}
                if device not in matrix_device_sums[matrix]:
                    matrix_device_sums[matrix][device] = [0.0, 0.0, 0.0]
                    matrix_device_counts[matrix][device] = 0

                matrix_device_sums[matrix][device][0] += wall_time
                matrix_device_sums[matrix][device][1] += mem_wall_time
                matrix_device_sums[matrix][device][2] += fault_time
                matrix_device_counts[matrix][device] += 1

except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit(1)

# Filter matrices with data for both CPU and GPU
filtered_matrices = [matrix for matrix in matrix_device_sums if 'CPU' in matrix_device_sums[matrix] and 'GPU' in matrix_device_sums[matrix]]

# Calculate maximum values for each matrix
matrix_max_values = {}
for matrix in matrix_device_sums:
    max_values = []
    for device in matrix_device_sums[matrix]:
        max_value = max(matrix_device_sums[matrix][device])
        max_values.append(max_value)
    matrix_max_values[matrix] = max(max_values)

# Sort matrices based on maximum values (from least to greatest)
sorted_matrices = sorted(matrix_max_values.keys(), key=lambda x: matrix_max_values[x])

# Extract data for plotting based on sorted matrices
matrix_names = sorted_matrices
devices = ['CPU', 'GPU']
num_matrices = len(matrix_names)
num_devices = len(devices)

# Create bar positions and width
bar_width = 0.35
index = np.arange(num_matrices)
bar_positions = [index + i * bar_width for i in range(num_devices)]

# Create the stacked bar graph with font size set to 16
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size as needed
colors = ['r', 'g', 'b']  # You can customize the colors

plt.yscale('log')
for i, device in enumerate(devices):
    avg_wall_times = [matrix_device_sums[matrix].get(device, [0.0, 0.0, 0.0])[0] / matrix_device_counts[matrix].get(device, 1) for matrix in matrix_names]
    avg_mem_wall_times = [matrix_device_sums[matrix].get(device, [0.0, 0.0, 0.0])[1] / matrix_device_counts[matrix].get(device, 1) for matrix in matrix_names]
    avg_fault_times = [matrix_device_sums[matrix].get(device, [0.0, 0.0, 0.0])[2] / matrix_device_counts[matrix].get(device, 1) for matrix in matrix_names]

    plt.bar(bar_positions[i], avg_wall_times, bar_width, label=f'{device} - PCG Compute Time', color=colors[i])
    if (device == 'GPU'):
        plt.bar(bar_positions[i], avg_mem_wall_times, bar_width, label=f'{device} - Memory Transfer Time', color=colors[i], alpha=0.5, bottom=avg_wall_times)
    if (device == 'CPU'):
        plt.bar(bar_positions[i], avg_fault_times, bar_width, label=f'{device} - Fault Checking Time', color=colors[i], alpha=0.2, bottom=np.array(avg_wall_times) + np.array(avg_mem_wall_times))

plt.xlabel('Matrix', fontsize=16)  # Set x-axis label font size
plt.ylabel('Average Time', fontsize=16)  # Set y-axis label font size
plt.title('Average Timings - Full Protection', fontsize=16)  # Set title font size
plt.xticks(index + bar_width * num_devices / 2, matrix_names, fontsize=16, rotation=45, ha='right')  # Set x-axis tick label font size
plt.yticks(fontsize=16)  # Set y-axis tick label font size
plt.legend(fontsize=16)  # Set legend font size

# Save the plot as an image file
plt.tight_layout()
filename = f"{csv_file.name}_plot.png"
plt.savefig(filename, dpi=800)
plt.clf()
