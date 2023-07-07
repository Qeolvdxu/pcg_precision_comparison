import csv
from tabulate import tabulate

def calculate_percentage(wall_time, mem_wall_time, fault_time):
    total_time = wall_time
    wall_time = wall_time - fault_time
    wall_time_percentage = (wall_time / total_time) * 100
    mem_wall_time_percentage = (mem_wall_time / total_time) * 100
    fault_time_percentage = (fault_time / total_time) * 100
    return wall_time_percentage, mem_wall_time_percentage, fault_time_percentage

# Replace 'input.csv' with the actual file name
filename = './combo.csv'

table_data = []

with open(filename, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        device = row['DEVICE']
        if device == 'CPU':  # Skipping CPU rows
            continue

        matrix = row['MATRIX']
        precision = row['PRECISION']
        iteration = int(row['ITERATIONS'])
        wall_time = float(row['WALL_TIME'])
        mem_wall_time = float(row['MEM_WALL_TIME'])
        fault_time = float(row['FAULT_TIME'])

        wall_time_percentage, mem_wall_time_percentage, fault_time_percentage = calculate_percentage(
            wall_time, mem_wall_time, fault_time)

        table_data.append([device, matrix, precision, iteration,
                           f"{wall_time_percentage:.2f}%", f"{mem_wall_time_percentage:.2f}%",
                           f"{fault_time_percentage:.2f}%"])

# Define the table headers
headers = ['Device', 'Matrix', 'Precision', 'Iteration',
           'Wall Time (%)', 'Mem Wall Time (%)', 'Fault Time (%)']

# Generate the table
table = tabulate(table_data, headers, tablefmt="fancy_grid")

# Print the table
print(table)

