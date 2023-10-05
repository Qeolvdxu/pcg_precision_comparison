import traceback
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to handle errors
def handle_error(error_message):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    line_number = exc_tb.tb_lineno
    print(f"Error (Line {line_number}): {error_message}")
    exit(1)  # Exit the program with an error code

try:
    # Read data from CSV file
    data = pd.read_csv('no_protection.csv')
except FileNotFoundError:
    handle_error("CSV file not found.")

# Check for missing or NaN values in the entire dataset
if data.isnull().any().any():
    missing_rows = data[data.isnull().any(axis=1)]
    print("Rows with missing or NaN values:")
    print(missing_rows)
    handle_error("CSV file contains missing or NaN values.")

# Drop columns after 'SLOW_DOWN'
try:
    data = data.loc[:, :'SLOW_DOWN']
except KeyError:
    handle_error("The 'SLOW_DOWN' column does not exist in the dataset.")

# Remove rows with SLOW_DOWN equal to 1 from the entire dataset
try:
    data = data[data['SLOW_DOWN'] != 1]
except KeyError:
    handle_error("The 'SLOW_DOWN' column does not exist in the dataset.")

# Drop rows with missing values in specific columns
try:
    data = data.dropna(subset=['ROW_2-NORM', 'SLOW_DOWN'])
except KeyError:
    handle_error("Columns 'ROW_2-NORM' or 'SLOW_DOWN' do not exist in the dataset.")

# Get unique MATRIX values
try:
    unique_matrices = data['MATRIX'].unique()
except KeyError:
    handle_error("The 'MATRIX' column does not exist in the dataset.")

if len(unique_matrices) == 0:
    handle_error("No unique 'MATRIX' values found in the dataset.")

# Loop over unique MATRIX values
for matrix_name in unique_matrices:
    try:
        # Filter data for the current MATRIX
        filtered_data = data[data['MATRIX'] == matrix_name]

        # Drop all rows with the maximum slowdown value
        max_slowdown = filtered_data['SLOW_DOWN'].max()
        filtered_data = filtered_data[filtered_data['SLOW_DOWN'] != max_slowdown]

        # Sort the data by 'ROW_2-NORM' for plotting
        filtered_data = filtered_data.sort_values(by='ROW_2-NORM')
        
        # Scatter plot
        plt.scatter(filtered_data['ROW_2-NORM'], filtered_data['SLOW_DOWN'], label='Data Points')
        
        # Polynomial regression
        degree = 2  # You can adjust the degree of the polynomial
        coefficients = np.polyfit(np.log10(filtered_data['ROW_2-NORM']), filtered_data['SLOW_DOWN'], degree)
        poly = np.poly1d(coefficients)
        
        # Generate x values for the curve (log scale)
        x_log_values = np.logspace(np.log10(filtered_data['ROW_2-NORM'].min()), np.log10(filtered_data['ROW_2-NORM'].max()), 100)
        
        # Calculate corresponding y values for the curve
        y_values = poly(np.log10(x_log_values))
        
        # Plot the polynomial regression curve
        plt.plot(x_log_values, y_values, label=f'Polynomial Degree {degree}', color='red')
        
        # Set logarithmic scale on x-axis
        plt.xscale('log')
        
        # Labels and legend
        plt.xlabel('ROW_2-NORM (log scale)', fontsize=16)
        plt.ylabel('SLOW_DOWN', fontsize=16)
        plt.legend(fontsize=16)
        
        # Add MATRIX name to the title
        plt.title(f'{matrix_name} row 2-norm and slowdown correlation ', fontsize=16)

        # Save the plot as an image file
        filename = f"{matrix_name}_plot.png"
        plt.savefig(filename, dpi=300)
        plt.clf()

    except Exception as e:
        handle_error(f"An error occurred for MATRIX {matrix_name}: {e}")

print("Program completed successfully.")
