# Static Visualization of predicted results
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings  # Import the warnings module
from matplotlib.patches import Ellipse
import re
import random

# Suppress the Matplotlib deprecation warning for get_cmap
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
# Define the colormap
cmap = plt.get_cmap('Greys')  # You can choose a different colormap here


# Function to find and load the file
def find_and_load_file(testing_file, scenario_id, agent_id, folder_path):
    with open(testing_file, 'r') as f:
        filenames = f.read().splitlines()

    for testing_filename in filenames:
        if scenario_id in testing_filename and agent_id in testing_filename:
            result_file_path = os.path.join(folder_path, testing_filename)
            if os.path.exists(result_file_path):
                input_data = np.load(result_file_path)
                return input_data
    return None


# specify the prediction result directory
source_directory = "<your_path>/predict_result"
destination_directory = "<your_path>/visualization_static/"
# The data from testing set is used to provide geometry information
testing_directory = "<your_path>/testing_processed"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Specify the filename text file of testing files
testing_file = "<your_path>/testing_filenames.txt"

"""
# Get the list of filenames in the directory
filenames = os.listdir(testing_directory)

# Write the filenames to the output file
with open(testing_file, 'w') as f:
    for filename in filenames:
        f.write(f"{filename}\n")

# Select files for illustration
# Initialize a dictionary to store files of different atypes
files_by_atype = {1: [], 2: [], 3: []}

# Iterate through files and classify them
for filename in os.listdir(source_directory):
    if filename.endswith('.npz'):
        # Extract atype from the filename
        atype = int(filename.split('__atype_')[1].split('.')[0])
        if atype in files_by_atype:
            files_by_atype[atype].append(filename)

"""

# This file contains a subset of testing files chosen specifically for illustration purposes.
result_file = "<your_path>/illustration_filenames.txt"
"""
with open(result_file, 'w') as f:
    # Extract 1% of files for each atype
    for atype, files in files_by_atype.items():
        sample_size = max(1, len(files) // 100)  # Select at least one file
        selected_files = random.sample(files, sample_size)
        
        # Write filenames to the text file
        for file in selected_files:
            f.write(f"{file}\n")
"""

# Read the generated file list
with open(result_file, 'r') as f:
    selected_files = [line.strip() for line in f]

for filename in selected_files:
    if filename.endswith(".npz"):
        file_path = os.path.join(source_directory, filename)
        data = np.load(file_path)

        """
            This part is to plot predicted trajectories and ground truth
        """
        probas = np.array(data['probas'])
        coordinates = np.array(data['coordinates'])
        covariance_matrices = np.array(data['covariance_matrices'])
        loss_coeff = data['loss_coeff']

        # Normalization as how trajectory data was processed in the prerender
        # For inD data
        gt = (data["target/future/xy"] - np.array([1.4715e+01, 4.3008e-03]))/10
        # For Waymo data
        # gt = (data["xy_future_gt"] - np.array([1.4715e+01, 4.3008e-03])) / 10
        valid_mask = data["target/future/valid"].flatten() == 1

        # Compute the log softmax along axis 1
        log_probs = np.log(np.exp(probas) / np.sum(np.exp(probas), axis=1, keepdims=True))

        # Now, log_probs contains log softmax values, and to get probabilities:
        probas_softmax = np.exp(log_probs)
        vmin = np.min(probas_softmax[0])
        vmax = np.max(probas_softmax[0])

        # Number of trajectories
        num_trajectories = coordinates.shape[1]

        # Sort trajectories based on probas_softmax values
        sorted_indices = np.argsort(probas_softmax[0])

        for i in sorted_indices:
            trajectory = coordinates[0, i, valid_mask, :]
            x = trajectory[:, 0]  # X-coordinates
            y = trajectory[:, 1]  # Y-coordinates

            # Use the probability for coloring the entire trajectory
            color = cmap((probas_softmax[0, i] - vmin) / (vmax - vmin))

            # Plot the trajectory with the colors
            plt.scatter(x, y, label=f'Predicted {i + 1}', c=[color], s=5)

            # Plot confidence ellipses
            # Specify the edge color with transparency
            edge_color = (0, 0, 1, 0.2)
            for j in range(0, len(x), 5):
                cov_matrix = covariance_matrices[0, i, j]
                cov_ellipse = Ellipse(xy=(x[j], y[j]), width=cov_matrix[0, 0], height=cov_matrix[1, 1],
                                       angle=np.degrees(np.arctan2(cov_matrix[1, 0], cov_matrix[0, 0])),
                                       edgecolor=edge_color, facecolor='none')
                plt.gca().add_patch(cov_ellipse)

        valid_mask_past = data["target/history/valid"].flatten() == 1
        valid_mask = np.concatenate((valid_mask_past, valid_mask))
        gt_past = (data["target/history/xy"] - np.array([1.4715e+01, 4.3008e-03]))/10
        gt = np.concatenate((gt_past, gt), axis=1)
        gt_x = gt[0, valid_mask, 0]
        gt_y = gt[0, valid_mask, 1]
        plt.scatter(gt_x, gt_y, label='Ground Truth', linestyle='--', color='red', s=7)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        """
            This part is to plot trajectories of other agents
        """
        # Find the corresponding input file to get the information of other agents
        match = re.match(r"pred_(\d+)__aid_(\d+)__atype_\d+\.npz", filename)
        if match:
            scenario_id, agent_id = match.groups()

        # Find and load the file
        input_data = find_and_load_file(testing_file, scenario_id, agent_id, testing_directory)
        other_future_xy_origin = input_data["other/future/xy"]
        other_future_xy = (other_future_xy_origin - np.array([1.4715e+01, 4.3008e-03]))/10
        other_past_xy_origin = input_data["other/history/xy"]
        other_past_xy = (other_past_xy_origin - np.array([1.4715e+01, 4.3008e-03]))/10
        other_xy = np.concatenate((other_past_xy, other_future_xy), axis=1)
        other_valid_past = input_data["other/history/valid"]
        other_valid_future = input_data["other/future/valid"]
        other_valid = np.concatenate((other_valid_past, other_valid_future), axis=1)

        other_type = input_data["other/agent_type"]

        # Determine the number of agents dynamically
        num_agents = other_future_xy.shape[0]

        for i in range(num_agents):
            other_valid_mask = other_valid[i, :, 0] == 1
            x = other_xy[i, other_valid_mask, 0]  # x coordinates
            y = other_xy[i, other_valid_mask, 1]  # y coordinates
            # Set different colors for different types of agents
            if other_type[i] == 1:
                other_color = "magenta"
            elif other_type[i] == 2:
                other_color = "orange"
            else:
                other_color = "green"

            plt.plot(x, y, marker='+', markersize=2, linestyle='-', linewidth=1, color=other_color, alpha=0.8)

        """
           This part is to plot the road network
       """
        road_segments = np.array(data["road_network_segments"])
        for line in road_segments:
            # Extract the coordinates of the two endpoints
            x1, y1 = (line[0] - np.array([1.4715e+01, 4.3008e-03])) / 10  # Endpoint 1
            x2, y2 = (line[1] - np.array([1.4715e+01, 4.3008e-03])) / 10  # Endpoint 2

            # Plot the line
            plt.plot([x1, x2], [y1, y2], color='black', linewidth=0.5,
                     alpha=0.5)  # Plot a line connecting the two points

        """
            This part is to setup other information of the plot itself
        """
        # Create a ScalarMappable object with the 'Greys' colormap
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('Greys'))

        # Pass this object to colorbar()
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Probability')
        cbar.set_ticks([0, 1])  # Set custom ticks at 0 and 1
        cbar.set_ticklabels([f'{vmin:.4f}', f'{vmax:.4f}'])  # Format vmin and vmax to have four decimal places

        # Check if the file name contains "aid_0" to set the overall label
        if "atype_1" in filename:
            overall_label = 'Vehicle'
        elif "atype_3" in filename:
            overall_label = 'Cyclist'
        else:
            overall_label = 'Pedestrian'

        # Add the overall label to the plot
        plt.annotate(overall_label, xy=(0.5, 0.95), xycoords='axes fraction', fontsize=12, ha='center')
        plt.legend()
        plt.grid(True)

        # Save the plot with a filename based on the input file
        plot_filename = os.path.splitext(filename)[0] + "_plot.png"
        plot_path = os.path.join(destination_directory, plot_filename)
        plt.savefig(plot_path)

        # Close the current plot to release resources
        plt.close()
