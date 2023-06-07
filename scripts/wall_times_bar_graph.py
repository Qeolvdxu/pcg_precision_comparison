import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cpuinfo
import GPUtil
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv('./combo.csv')

# Filter out rows with missing values
df = df.dropna()

# Filter rows for CPU and GPU devices separately
df_cpu = df[df['DEVICE'] == 'CPU']
df_gpu = df[df['DEVICE'] == 'GPU']

# Group the CPU data by MATRIX and calculate the mean of WALL_TIME, FAULT_TIME, and ITERATIONS
df_cpu_grouped = df_cpu.groupby('MATRIX')[['WALL_TIME', 'FAULT_TIME', 'ITERATIONS']].mean()

# Group the GPU data by MATRIX and calculate the mean of WALL_TIME, FAULT_TIME, and ITERATIONS
df_gpu_grouped = df_gpu.groupby('MATRIX')[['WALL_TIME', 'FAULT_TIME', 'ITERATIONS']].mean()

# Group the data by MATRIX and calculate the mean of MEM_WALL_TIME, FAULT_TIME, and ITERATIONS
df_mem_grouped = df.groupby('MATRIX')[['MEM_WALL_TIME', 'FAULT_TIME', 'ITERATIONS']].mean()

# Combine the three data series into a single DataFrame
df_combined = pd.DataFrame({'CPU Wall Time': df_cpu_grouped['WALL_TIME'],
                            'GPU Wall Time': df_gpu_grouped['WALL_TIME'],
                            'Mem Wall Time': df_mem_grouped['MEM_WALL_TIME'],
                            'Fault Time': df_mem_grouped['FAULT_TIME'],
                            'Iteration Count': df_mem_grouped['ITERATIONS']})

# Sort the DataFrame by the sum of WALL_TIME, MEM_WALL_TIME, and FAULT_TIME in descending order
df_combined_sorted = df_combined.sort_values(by=['Iteration Count'], ascending=False)

# Calculate the GPU Wall Time by subtracting the Fault Time from the total Wall Time
df_combined_sorted['GPU Wall Time'] -= df_combined_sorted['Fault Time']

# Get the matrix names in the sorted order
matrix_names = df_combined_sorted.index.tolist()
matrix_names = [name.replace('../../test_subjects/norm/', '') for name in matrix_names]

# Get the number of matrices
num_matrices = len(matrix_names)

# Set the bar width
bar_width = 0.2

# Get the CPU model name using cpuinfo
cpu_info = cpuinfo.get_cpu_info()
cpu_name = cpu_info['brand_raw']

# Get the GPU model name using GPUtil
gpus = GPUtil.getGPUs()
if len(gpus) > 0:
    gpu_name = gpus[0].name
else:
    gpu_name = 'N/A'

# Set the dark theme style with low contrast using Seaborn
sns.set_style("darkgrid")
sns.set_palette("dark")

# Plotting for Wall Time
fig, ax = plt.subplots()

# Plot the memory wall time bars
mem_bar = ax.bar(matrix_names, df_combined_sorted['Mem Wall Time'], width=bar_width, label='Mem Wall Time')

# Plot the fault time bars
fault_bar = ax.bar(matrix_names, df_combined_sorted['Fault Time'], width=bar_width,
                   bottom=df_combined_sorted['Mem Wall Time'], label='Fault Time')

# Plot the GPU wall time bars
gpu_bar = ax.bar(matrix_names, df_combined_sorted['GPU Wall Time'], width=bar_width,
                 bottom=df_combined_sorted['Mem Wall Time'] + df_combined_sorted['Fault Time'],
                 label='GPU Wall Time')

# Set the x-axis label
ax.set_xlabel('Matrix')

# Set the y-axis label
ax.set_ylabel('Time (ms)')

# Set the title
ax.set_title('Wall Time Comparison')

# Add a legend
ax.legend()
ax.set_yscale("log")

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=90)

# Save the Wall Time plot as SVG
plt.tight_layout()
plt.savefig('wall_time_plot.svg', format='svg')
plt.close()

# Plotting for Iteration Count
fig, ax = plt.subplots()

# Plot the iteration count bars
iteration_bar = ax.bar(matrix_names, df_combined_sorted['Iteration Count'], width=bar_width, label='Iteration Count')

# Set the x-axis label
ax.set_xlabel('Matrix')

# Set the y-axis label
ax.set_ylabel('Iteration Count')

# Set the title
ax.set_title('Iteration Count Comparison')

# Add a legend
ax.legend()

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=90)

# Save the Iteration Count plot as SVG
plt.tight_layout()
plt.savefig('iteration_count_plot.svg', format='svg')
plt.close()

