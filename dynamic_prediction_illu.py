# illustrate the predicted trajectory and safety status in the roadmap
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse
import re

def find_and_load_file(input_filenames, scenario_id, folder_path):
    with open(input_filenames, 'r') as f:
        filenames = f.read().splitlines()

    for input_filename in filenames:
        if scenario_id in input_filename:
            input_file_path = os.path.join(folder_path, input_filename)
            if os.path.exists(input_file_path):
                input_data = np.load(input_file_path)
                return input_data
    return None

def create_vehicle_animation_prediction(ax, vehicle_data, prediction_info):
    # Create a scatter plot object for predicted trajectories
    prediction_points = prediction_info['coordinates']
    colors = prediction_info['color_prediction']
    points_scatters = [ax.scatter([], [], color=colors[k], s=9, visible=False) for k in range(prediction_points.shape[0])]
    prediction_point_valid = prediction_info['valid_mask_prediction']
    sorted_indices = prediction_info['sorted_indices']

    # create an array to store the target vehicle rectangle
    target_color = 'red'  # default color for target agent
    target_xy = prediction_info["gt"]
    # in the Rectangle, the angle should be in radians
    # in 'prediction_info', yaw is in radians
    target_rect = Rectangle(target_xy[0], prediction_info["target/length"], prediction_info["target/width"],
                            angle=prediction_info["yaw"][0], facecolor=None, edgecolor=target_color)
    target_valid = prediction_info["valid_mask_gt"]
    ax.add_patch(target_rect)


    # create an array to store other vehicle rectangles
    vehicles = []
    text_labels = []  # Initialize text labels list

    # create vehicle rectangles and draw on axis
    for i in range(len(vehicle_data)):
        # find the xy-limits of road user movements
        veh_data_i = vehicle_data[f"dict_{i}"]

        if veh_data_i["current/length"] < 1e-6:
            veh_data_i["current/length"]

        if veh_data_i["valid"][0] == 1:
            rect_color = 'blue'  # default color for other agents
            if veh_data_i["type"] == 2:  # differentiate between agent types
                rect_color = (0.12890625, 0.5703125, 0.19140625)
            elif veh_data_i["type"] == 3:
                rect_color = (0.99609375, 0.5859375, 0.09375)
            rect = Rectangle(veh_data_i["xy"][0], veh_data_i["current/length"], veh_data_i["current/width"],
                             angle=veh_data_i["bbox_yaw"][0], facecolor=None, edgecolor=rect_color)
        else:
            rect = Rectangle((0, 0), 0, 0, angle=0)

        ax.add_patch(rect)
        vehicles.append(rect)

        # Add text label for the ID
        offset = 2
        text = ax.text(veh_data_i["xy"][0][0], veh_data_i["xy"][0][1] + offset, str(i), ha='center', va='center', fontsize=4)
        text_labels.append(text)  # Append text label to the list

    # update the location and angle of the agent rectangles
    def update(frame):
        artists = []

        # update target vehicle
        if target_valid[frame]:
            target_rect.set_visible(True)
            target_rect.set_xy(target_xy[frame])
            target_rect.set_angle(prediction_info["yaw"][frame])
            target_rect.set_width(prediction_info["target/length"])
            target_rect.set_height(prediction_info["target/width"])
            target_rect.set_color(target_color)  # Ensure color is set if needed
        else:
            target_rect.set_visible(False)
        artists.append(target_rect)

        # update other vehicles
        for i, rect in enumerate(vehicles):
            veh_data_i = vehicle_data[f"dict_{i}"]
            if veh_data_i["valid"][frame] == 1:
                rect.set_visible(True)
                rect.set_xy(veh_data_i["xy"][frame])
                rect.set_width(veh_data_i["current/length"])
                rect.set_height(veh_data_i["current/width"])

                # Rotate the rectangle by the angle
                rect.set_angle(veh_data_i["bbox_yaw"][frame])
                # Update text position
                text_labels[i].set_position(veh_data_i["xy"][frame])

            else:
                # If data is not valid, hide the rectangle
                rect.set_visible(False)
                text_labels[i].set_visible(False)
            artists.append(rect)
            artists.append(text_labels[i])

        # update predicted trajectories

        if frame > 10:
            for k, scatter in enumerate(points_scatters):
                if prediction_point_valid[frame]:
                    # Collect all points up to the current frame
                    points_to_plot = prediction_points[sorted_indices[k], :frame - 10, :]
                    scatter.set_offsets(points_to_plot)
                    scatter.set_visible(True)
                else:
                    scatter.set_visible(False)
                artists.append(scatter)

        return artists

    # create the animation
    ani = FuncAnimation(fig, update, frames=range(target_xy.shape[0]), interval=100, blit=True)

    return ani


source_directory = "/home/meya174e/bin/inD_data/model_0802/predict_result"
destination_directory = "/home/meya174e/bin/inD_data/model_0802/visualization_dynamic/"
testing_directory = "/home/meya174e/bin/inD_data/model_0802/testing_processed"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Specify the filename text file of testing files
testing_file = "/home/meya174e/bin/inD_data/model_0802/testing_filenames.txt"

# write the selected filenames into a txt file
result_file = "/home/meya174e/bin/inD_data/model_0802/illustration_filenames.txt"

#with open(input_file, 'w') as f:
#    for filename in os.listdir(input_directory):
#        f.write(f"{filename}\n")

cmap = plt.get_cmap('viridis_r')  # You can choose a different colormap here

with open(result_file, 'r') as f:
    selected_files = [line.strip() for line in f]

for filename in selected_files[:10]:
    if filename.endswith(".npz"):
        file_path = os.path.join(source_directory, filename)
        data = np.load(file_path)

        vehicle_data = {}
        prediction_info = {}
        """
            this part is to plot predicted trajectories and ground truth
        """
        probas = np.array(data['probas'])
        coordinates = np.array(data['coordinates']) * 10
        covariance_matrices = np.array(data['covariance_matrices'])
        loss_coeff = data['loss_coeff']

        # normalization as what how trajectory data was processed in the prerender
        # for inD data
        gt = (data["target/future/xy"] - np.array([1.4715e+01, 4.3008e-03]))
        # for Waymo data
        # gt = (data["xy_future_gt"] - np.array([1.4715e+01, 4.3008e-03]))
        valid_mask = data["target/future/valid"].flatten() == 1

        # Compute the log softmax along axis 1
        log_probs = np.log(np.exp(probas) / np.sum(np.exp(probas), axis=1, keepdims=True))

        # Now, log_probs contains log softmax values, and to get probabilities:
        probas_softmax = np.exp(log_probs)
        probas_min = np.min(probas_softmax[0])
        probas_max = np.max(probas_softmax[0])

        # Number of trajectories
        num_trajectories = coordinates.shape[1]

        # Sort trajectories based on probas_softmax values
        sorted_indices = np.argsort(probas_softmax[0])

        # Initialize colors based on probas_softmax
        colors = [cmap((probas_softmax[0, k] - probas_min) / (probas_max - probas_min)) for k in range(num_trajectories)]

        valid_mask_past = data["target/history/valid"].flatten() == 1
        valid_mask = np.concatenate((valid_mask_past, valid_mask))
        gt_past = (data["target/history/xy"] - np.array([1.4715e+01, 4.3008e-03]))
        gt = np.concatenate((gt_past, gt), axis=1)
        target_current_yaw = data["current_direction"]
        print(target_current_yaw)
        angle_traj = np.concatenate((data["target/history/yaw"][0], data["target/future/yaw"][0])) + target_current_yaw

        valid_mask_prediction = valid_mask.copy()
        # Set the first 11 elements to False
        valid_mask_prediction[:11] = False

        prediction_info = ({
            'coordinates': coordinates[0],
            'cov_matrix': covariance_matrices[0],
            'sorted_indices': sorted_indices,
            'gt': gt[0],
            'valid_mask_gt': valid_mask,
            'valid_mask_prediction': valid_mask_prediction,
            'color_prediction': colors,
            'target/length': data["target/length"].item(),
            'target/width': data["target/width"].item(),
            'yaw': angle_traj.flatten()
        })

        """
            this part is to collect trajectories of other agents
        """
        agents_data = {}
        # find the corresponding input file to get the information of other agents
        match = re.search(r'pred_(\d+)__aid', filename)
        if match:
            scenario_id = match.group(1)

        # Find and load the file
        # here the data are prerendered data
        input_data = find_and_load_file(testing_file, scenario_id, testing_directory)
        other_future_xy_origin = input_data["other/future/xy"]
        other_future_xy = (other_future_xy_origin - np.array([1.4715e+01, 4.3008e-03]))
        other_past_xy_origin = input_data["other/history/xy"]
        other_past_xy = (other_past_xy_origin - np.array([1.4715e+01, 4.3008e-03]))
        other_xy = np.concatenate((other_past_xy, other_future_xy), axis=1)
        other_valid_past = input_data["other/history/valid"]
        other_valid_future = input_data["other/future/valid"]
        other_valid = np.concatenate((other_valid_past, other_valid_future), axis=1)
        other_type = input_data["other/agent_type"]
        other_yaw = np.concatenate((input_data["other/history/yaw"], input_data["other/future/yaw"]), axis=1) + target_current_yaw
        # other_yaw = np.concatenate((input_data["other/history/yaw"], input_data["other/future/yaw"]), axis=1)
        print(other_yaw[-1, :, :])
        print(other_yaw.shape)
        # Determine the number of agents dynamically
        num_agents = other_future_xy.shape[0]

        for i in range(num_agents):
            other_valid_mask = other_valid[i, :, 0] == 1
            x = other_xy[i, other_valid_mask, 0]  # x coordinates
            y = other_xy[i, other_valid_mask, 1]  # y coordinates
            # set different colors for different types of agents
            if other_type[i] == 1:
                other_color = "magenta"
            elif other_type[i] == 2:
                other_color = "orange"
            else:
                other_color = "green"

            agent_data_i = ({
                'xy': other_xy[i],
                'bbox_yaw': other_yaw[i],
                'valid': other_valid[i],
                'type': other_type[i],
                'current/length': input_data["other/length"][i].item(),
                'current/width': input_data["other/width"][i].item()
            })

            key = f"dict_{i}"
            agents_data[key] = agent_data_i

        # arrange the canvas
        fig, ax = plt.subplots()

        """
           this part is to plot the road network
        """
        road_segments = np.array(data["road_network_segments"])

        # Plot static data (intersection geometry lines)
        # Initialize min and max values for x and y
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for line in road_segments:
            # Extract the coordinates of the two endpoints
            x1, y1 = (line[0] - np.array([1.4715e+01, 4.3008e-03]))  # Endpoint 1
            x2, y2 = (line[1] - np.array([1.4715e+01, 4.3008e-03]))   # Endpoint 2

            # Update the min and max values
            min_x = min(min_x, x1, x2)
            max_x = max(max_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_y = max(max_y, y1, y2)

            # Plot the line
            plt.plot([x1, x2], [y1, y2], color='black', linewidth=0.5,
                     alpha=0.5)  # Plot a line connecting the two points

        ani = create_vehicle_animation_prediction(ax, agents_data, prediction_info)

        ax.set_aspect('equal')
        ax.set_xlabel('X/m')
        ax.set_ylabel('Y/m')
        # Set the plot limits based on the min and max values
        ax.set_xlim(min_x - 10, max_x + 10)  # Add some padding
        ax.set_ylim(min_y - 10, max_y + 10)  # Add some padding

        # Modify the tick labels to show 10 times the current values
        #def change_tick_labels(tick_val, pos):
        #    return f'{tick_val * 10:.1f}'

        #ax.xaxis.set_major_formatter(plt.FuncFormatter(change_tick_labels))
        #ax.yaxis.set_major_formatter(plt.FuncFormatter(change_tick_labels))

        """
            this part is to setup other information of the plot itself
        """
        # create a ScalarMappable object with the colormap
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis_r'))

        # pass this object to colorbar()
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Probability')
        cbar.set_ticks([0, 1])  # Set custom ticks at 0 and 1
        cbar.set_ticklabels([f'{probas_min:.4f}', f'{probas_max:.4f}'])  # Format vmin and vmax to have four decimal places

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
        plot_filename = os.path.splitext(filename)[0] + "_plot.gif"
        plot_path = os.path.join(destination_directory, plot_filename)
        ani.save(plot_path, writer='imagemagick', fps=3)

        # Close the current plot to release resources
        plt.close()






