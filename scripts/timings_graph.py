import matplotlib.pyplot as plt
import numpy as np

# Data from the CSV file
# Update the variables with the appropriate column indices based on your data
device_col = 0
matrix_col = 1
wall_time_col = 4
mem_time_col = 5
fault_time_col = 6

# Extract relevant data from the CSV file
devices = []
matrices = []
wall_times_cpu = []
wall_times_gpu = []
mem_times_gpu = []
fault_times_gpu = []

with open('Data/combo.csv', 'r') as file:
    next(file)  # Skip the header row
    for line in file:
        row = line.strip().split(',')
        devices.append(row[device_col])
        matrices.append(row[matrix_col])
        wall_time = float(row[wall_time_col])
        mem_time = float(row[mem_time_col])
        fault_time = float(row[fault_time_col])
        if row[device_col] == 'CPU':
            wall_times_cpu.append(wall_time)
        else:
            wall_times_gpu.append(wall_time-fault_time)
            mem_times_gpu.append(mem_time)
            fault_times_gpu.append(fault_time)

# Calculate the number of devices and matrices
num_devices = len(set(devices))
num_matrices = len(set(matrices))

# Set the bar positions
bar_width = 0.35
index = np.arange(num_matrices)*2


# Create the bar graph
plt.bar(index + (bar_width)*0, wall_times_cpu, bar_width, label='CPU WALL')
plt.bar(index + (bar_width)*1, mem_times_gpu, bar_width, label='GPU MEM')
plt.bar(index + (bar_width)*1, fault_times_gpu, bar_width,bottom=mem_times_gpu, label='GPU FAULT')
plt.bar(index + (bar_width)*1, wall_times_gpu, bar_width, bottom=np.array(mem_times_gpu) + np.array(fault_times_gpu), label='GPU WALL')

plt.yscale('log',base=10)
plt.subplots_adjust(bottom=0.5)

# Add labels, titles, and legend
plt.xlabel('Matrix')
plt.ylabel('Wall Time')
plt.title('Comparison of Wall Time for CPU and GPU')
plt.xticks(index, set(matrices), rotation=45)
plt.legend()


plt.savefig('Data/timings.svg')
# Display the graph
plt.show()
