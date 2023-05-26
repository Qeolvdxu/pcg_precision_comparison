import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('../src/build/combo.csv')

# Filter out rows with missing values
df = df.dropna()

# Filter rows for CPU and GPU devices separately
df_cpu = df[df['DEVICE'] == 'CPU']
df_gpu = df[df['DEVICE'] == 'GPU']

# Group the CPU data by MATRIX and calculate the mean of WALL_TIME
df_cpu_grouped = df_cpu.groupby('MATRIX')['WALL_TIME'].mean()

# Group the GPU data by MATRIX and calculate the mean of WALL_TIME
df_gpu_grouped = df_gpu.groupby('MATRIX')['WALL_TIME'].mean()

# Group the data by MATRIX and calculate the mean of MEM_WALL_TIME
df_mem_grouped = df.groupby('MATRIX')['MEM_WALL_TIME'].mean()

# Combine the three data series into a single DataFrame
df_combined = pd.DataFrame({'CPU Wall Time': df_cpu_grouped,
                            'GPU Wall Time': df_gpu_grouped,
                            'Mem Wall Time': df_mem_grouped})

# Get the matrix names
matrix_names = df_combined.index.tolist()
matrix_names = [name.replace('../../test_subjects/norm/', '') for name in matrix_names]

# Get the number of matrices
num_matrices = len(matrix_names)

# Set the bar width
bar_width = 0.3

# Set the positions of the bars on the x-axis
index = np.arange(num_matrices)

# Plotting
fig, ax = plt.subplots()
mem_bar = ax.bar(index + (2 * bar_width), df_combined['Mem Wall Time'], bar_width, label='Mem Wall Time')
print(df_combined['CPU Wall Time'])
cpu_bar = ax.bar(index, df_combined['CPU Wall Time'], bar_width, label='CPU Wall Time')
gpu_bar = ax.bar(index + bar_width, df_combined['GPU Wall Time'], bar_width, label='GPU Wall Time')



# Set the x-axis labels to matrix names
ax.set_xticks(index + bar_width)
ax.set_xticklabels(matrix_names, rotation=45)

# Set the y-axis label
ax.set_ylabel('Time (ms)')

# Set the title
ax.set_title('CPU vs GPU Wall Time and Mem Wall Time')

# Add a legend
ax.legend()

ax.set_yscale("log")

# Show the plot
plt.tight_layout()
# Save the graph as SVG
plt.savefig('graph.svg', format='svg')
plt.show()
