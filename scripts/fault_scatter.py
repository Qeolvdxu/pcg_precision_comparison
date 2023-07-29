import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("scatter-test.csv", sep=",")

# Strip whitespaces from column names
data.columns = data.columns.str.strip()

# Group the data by unique 'MATRIX' values
grouped_data = data.groupby('MATRIX')

# Define color mapping for CPU and GPU
color_map = {'CPU': 'blue', 'GPU': 'green'}

# Create a separate scatter plot for each unique 'MATRIX'
for matrix, group in grouped_data:
    row_2_norm = group['ROW_2-NORM']
    wall_time = group['WALL_TIME']
    device = group['DEVICE']

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    for i in range(len(device)):
        plt.scatter(row_2_norm.iloc[i], wall_time.iloc[i], s=50, alpha=0.7, c=color_map.get(device.iloc[i], 'blue'), edgecolors='k')

    # Add labels and title
    plt.xlabel('Row 2-Norm')
    plt.ylabel('Wall Time')
    plt.title(f'Scatter Plot for MATRIX: {matrix}')

    # Show the plot
    plt.grid(True)
    plt.show()

