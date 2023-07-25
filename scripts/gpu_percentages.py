import csv
import svgwrite

def calculate_percentage(wall_time, mem_wall_time, fault_time):
    total_time = wall_time + fault_time + mem_wall_time
    wall_time_percentage = (wall_time / total_time) * 100
    mem_wall_time_percentage = (mem_wall_time / total_time) * 100
    fault_time_percentage = (fault_time / total_time) * 100
    return wall_time_percentage, mem_wall_time_percentage, fault_time_percentage

# Replace 'input.csv' with the actual file name
filename = 'Data/combo.csv'

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

# Convert table to SVG
dwg = svgwrite.Drawing('Data/gpu_table.svg', profile='tiny')
table_font_size = 12
cell_width = 120
cell_height = 30

# Set table background color
background_color = '#F5F5F5'

# Draw table background
dwg.add(dwg.rect(insert=(0, 0), size=(len(headers) * cell_width, (len(table_data) + 1) * cell_height),
                 fill=background_color))

# Draw table headers
for i, header in enumerate(headers):
    dwg.add(dwg.rect(insert=(i * cell_width, 0), size=(cell_width, cell_height), fill='gray', stroke='black'))
    dwg.add(dwg.text(header, insert=(i * cell_width + 5, cell_height / 2), fill='white', font_size=table_font_size))

# Draw table cells
for row_index, row in enumerate(table_data):
    for col_index, cell in enumerate(row):
        dwg.add(dwg.rect(insert=(col_index * cell_width, (row_index + 1) * cell_height),
                         size=(cell_width, cell_height), fill='none', stroke='black'))
        dwg.add(dwg.text(cell, insert=(col_index * cell_width + 5, (row_index + 2) * cell_height - 5),
                         fill='black', font_size=table_font_size))

# Save the SVG file
dwg.save()
