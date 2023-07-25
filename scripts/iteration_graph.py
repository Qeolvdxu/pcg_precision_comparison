import matplotlib.pyplot as plt
import numpy as np
import csv

# Data from the CSV file
# Update the variables with the appropriate column indices based on your data
device_col = 0
matrix_col = 1
iter_col = 3

# Extract relevant data from the CSV file
matrix_data = {}

with open('Data/combo.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        matrix = row[matrix_col]
        iter_val = float(row[iter_col])
        if matrix not in matrix_data:
            matrix_data[matrix] = {'CPU': 0, 'GPU': 0}
        if row[device_col] == 'CPU':
            matrix_data[matrix]['CPU'] += iter_val
        else:
            matrix_data[matrix]['GPU'] += iter_val

# Extract sorted matrix names, CPU iteration values, and GPU iteration values
sorted_matrices = []
sorted_cpu_iter = []
sorted_gpu_iter = []

for matrix, iter_values in matrix_data.items():
    sorted_matrices.append(matrix)
    sorted_cpu_iter.append(iter_values['CPU'])
    sorted_gpu_iter.append(iter_values['GPU'])

# Calculate the number of matrices
num_matrices = len(sorted_matrices)

# Set the bar positions
bar_width = 0.35
index = np.arange(num_matrices)

# Create the bar graph
plt.bar(index, sorted_cpu_iter, bar_width, label='CPU WALL')
plt.bar(index + bar_width, sorted_gpu_iter, bar_width, label='GPU MEM')

plt.subplots_adjust(bottom=0.5)

# Add labels, titles, and legend
plt.xlabel('Matrix')
plt.ylabel('Iterations')
plt.title('Iteration Comparison of CPU and GPU')
plt.xticks(index + bar_width / 2, sorted_matrices, rotation=45)
plt.legend()

plt.savefig('Data/iterations.svg')
# Display the graph
plt.show()

