import matplotlib.pyplot as plt
import numpy as np

# Data from the CSV file
# Update the variables with the appropriate column indices based on your data
device_col = 0
matrix_col = 1
iter_col = 3

# Extract relevant data from the CSV file
devices = []
matrices = []

gpu_iter = []
cpu_iter = []

with open('Data/combo.csv', 'r') as file:
    next(file)  # Skip the header row
    for line in file:
        row = line.strip().split(',')
        devices.append(row[device_col])
        matrices.append(row[matrix_col])
        iter = float(row[iter_col])
        if row[device_col] == 'CPU':
            cpu_iter.append(iter)
        else:
            gpu_iter.append(iter)

# Calculate the number of devices and matrices
num_devices = len(set(devices))
num_matrices = len(set(matrices))

# Set the bar positions
bar_width = 0.35
index = np.arange(num_matrices)*2


# Create the bar graph
plt.bar(index + (bar_width)*0, cpu_iter, bar_width, label='CPU WALL')
plt.bar(index + (bar_width)*1, gpu_iter, bar_width, label='GPU MEM')

plt.subplots_adjust(bottom=0.5)

# Add labels, titles, and legend
plt.xlabel('Matrix')
plt.ylabel('Iterations')
plt.title('Iteration Comparison of CPU and GPU')
plt.xticks(index, set(matrices), rotation=45)
plt.legend()


plt.savefig('Data/iterations.svg')
# Display the graph
plt.show()
