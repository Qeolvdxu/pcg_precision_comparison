import matplotlib.pyplot as plt
import numpy as np

def create_donut_chart(labels, values):
    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Customize the text properties
    text_props = {'fontsize': 12, 'color': 'black', 'fontweight': 'bold'}

    # Calculate the total value
    total = sum(values)

    # Convert values to percentages
    percentages = [(value / total) * 100 for value in values]

    # Draw the outer pie chart
    outer_wedges, _, outer_autopct = ax.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90, textprops=text_props)

    # Customize the outer wedge properties
    outer_wedge_props = {'edgecolor': 'white', 'linewidth': 1}

    for wedge in outer_wedges:
        wedge.set(**outer_wedge_props)

    # Draw the inner white circle to create a donut chart
    inner_circle = plt.Circle((0, 0), 0.5, color='white')
    ax.add_artist(inner_circle)

    # Set equal aspect ratio to draw a circle
    ax.axis('equal')

    # Set the title for the donut chart
    ax.set_title("Donut Chart", fontsize=16, fontweight='bold')

    # Create a legend with custom formatting
    legend_labels = [f'{label}: {value} ({percentage:.1f}%)' for label, value, percentage in zip(labels, values, percentages)]
    legend = ax.legend(outer_wedges, legend_labels, title="Legend", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.setp(legend.get_title(), fontsize=12, fontweight='bold')

    # Set the y-axis scale to logarithmic
    ax.set_yscale('log')

    # Show the donut chart
    plt.show()

# Gather user input
num_slices = int(input("Enter the number of data slices: "))

labels = []
values = []

# Get input for each slice
for i in range(1, num_slices + 1):
    label = input(f"Enter label for slice {i}: ")
    value = float(input(f"Enter value for slice {i}: "))
    
    labels.append(label)
    values.append(value)

# Create the donut chart
create_donut_chart(labels, values)

