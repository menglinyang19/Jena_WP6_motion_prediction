# This script is used to predict the future trajectory of agents in a given scenario using a trained model.
import torch
from model.multipathpp import MultiPathPP
from prerender.utils.utils import get_config
from model.data import get_dataloader, dict_to_cuda, normalize
#from model.data import get_dataloader, dict_to_cpu, normalize
from tqdm import tqdm
import numpy as np
import os
import traceback
from itertools import islice

class AgentFilter:
    def _get_only_fully_available_agents(self, data):
        future_validity = data["target/future/valid"]
        n_valid_timestamps = future_validity.sum().item()
        return n_valid_timestamps

    def _get_trajectory_class(self, data):
        # to generate 'trajectory_bucket'
        valid = np.concatenate(
            [data["target/history/valid"][0, -1:, 0], data["target/future/valid"][0, :, 0]])
        future_xy = np.concatenate(
            [data["target/history/xy"][0, -1:, :], data["target/future/xy"][0, :, :]])
        future_yaw = np.concatenate(
            [data["target/history/yaw"][0, -1:, 0], data["target/future/yaw"][0, :, 0]])
        future_speed = np.concatenate(
            [data["target/history/speed"][0, -1:, 0], data["target/future/speed"][0, :, 0]])

        kMaxSpeedForStationary = 1.0                 # (m/s)
        kMaxDisplacementForStationary = 5.0          # (m)
        kMaxLateralDisplacementForStraight = 5.0     # (m)
        kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
        kMaxAbsHeadingDiffForStraight = np.pi / 6.0   # (rad)
        first_valid_index, last_valid_index = 0, None
        # check the valid range
        for i in range(1, len(valid)):
            if valid[i] == 1:
                last_valid_index = i
        if valid[first_valid_index] == 0 or last_valid_index is None:
            return None

        xy_delta = future_xy[last_valid_index] - future_xy[first_valid_index]
        final_displacement = np.linalg.norm(xy_delta)      # euclidean distance
        heading_delta = future_yaw[last_valid_index] - future_yaw[first_valid_index]
        max_speed = max(future_speed[last_valid_index], future_speed[first_valid_index])

        if max_speed < kMaxSpeedForStationary and \
                final_displacement < kMaxDisplacementForStationary:
            return "stationary"
        if np.abs(heading_delta) < kMaxAbsHeadingDiffForStraight:
            if np.abs(xy_delta[1]) < kMaxLateralDisplacementForStraight:
                return "straight"
            return "straight_right" if xy_delta[1] < 0 else "straight_left"
        if heading_delta < -kMaxAbsHeadingDiffForStraight and xy_delta[1]:
            return "right_u_turn" if xy_delta[0] < kMinLongitudinalDisplacementForUTurn \
                else "right_turn"
        if xy_delta[0] < kMinLongitudinalDisplacementForUTurn:
            return "left_u_turn"
        return "left_turn"

# Specify the path to the .pth file
file_path = "/home/meya174e/bin/Multipath++/inD_data/last.pth" # trained model

config_path = "/home/meya174e/bin/Multipath++/inD_data/configs/final_RoP_Cov_A_fMCG.yaml"
config = get_config(config_path)

model = MultiPathPP(config["model"])
# Set the device to CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to("cpu")  # Move the model to the CPU
    
# Move the model to the device
model.to(device)

# Access the loaded objects
model_state_dict = torch.load(file_path, map_location=device)["model_state_dict"]
model.load_state_dict(model_state_dict)

num_steps = torch.load(file_path, map_location=torch.device('cpu'))["num_steps"]

dataloader = get_dataloader(config["predict"]["data_config"])
pbar = tqdm(dataloader)

# Create a directory to store the NPZ files
output_dir = "/home/meya174e/bin/inD_data/model_0802/predict_result"
os.makedirs(output_dir, exist_ok=True)

# Open the error log file in append mode
error_log_file = "/home/meya174e/bin/inD_data/model_0802/error_predict.txt"
with open(error_log_file, "a") as error_log:

#for i, data in enumerate(islice(pbar, 500)):
    for i, data in enumerate(pbar):

        # Extract relevant information from the input data
        scenario_id = '_'.join(str(item) for item in data['scenario_id'])
        agent_id = '_'.join(str(item) for item in data['agent_id'])
        agent_type = int(data['target/agent_type'].item())

        agent_filter_instance = AgentFilter()  # Create an instance of the AgentFilter class
        n_valid_timestamps  = agent_filter_instance._get_only_fully_available_agents(data)  # Call the method
        trajectory_bucket = agent_filter_instance._get_trajectory_class(data)

        try: 
            # Construct the output file name
            output_file_name = f"pred_{scenario_id}__aid_{agent_id}__atype_{agent_type}.npz"
            #   time_str = f"{current_time:07d}"
            #   location_str = f"{location_id}"
            #   track_str = f"{i:02d}"
            #   scenario_id = time_str + location_str + track_str

            # the following are prerender data
            with torch.no_grad():
                # extract current speed of the target agent
                current_speed = data['target/history/speed'].cpu().numpy()[:, -1, :]

                selected_input_data = {
                    'current_position': data['shift'].cpu().numpy(),
                    'current_direction': data['yaw'].cpu().numpy(),
                    'target/width': data['target/width'],
                    'target/length': data['target/length'],
                    'target/history/xy': data['target/history/xy'].cpu().numpy(),
                    'target/history/yaw': data['target/history/yaw'].cpu().numpy(),
                    'target/history/speed': data['target/history/speed'].cpu().numpy(),
                    'target/history/valid': data["target/history/valid"].cpu().numpy(),
                    'target/future/xy': data['target/future/xy'].cpu().numpy(),
                    'target/future/yaw': data['target/future/yaw'].cpu().numpy(),
                    'target/future/speed': data['target/future/speed'].cpu().numpy(),
                    'target/future/valid': data["target/future/valid"].cpu().numpy(),
                    'road_network_embeddings': data["road_network_embeddings"].cpu().numpy(),
                    'road_network_segments': data["road_network_segments"].cpu().numpy()
                    # Those are already transferred into the agent-based coordination
                }
                
                selected_input_data = {key: torch.tensor(value).to(device) for key, value in selected_input_data.items()}

                xy_future_gt = data["target/future/xy"]
                # to normalize the ground truth
                if config["train"]["normalize_output"]:
                    normalization_params = torch.Tensor([1.4715e+01, 4.3008e-03])
                    xy_future_gt = (data["target/future/xy"] - normalization_params) / 10.
                
                # Forward pass through the model
                if config["train"]["normalize"]:
                    data = normalize(data, config)
                #dict_to_cuda(data)
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(device)
                    
                probas, coordinates, covariance_matrices, loss_coeff = model(data, num_steps)

        except AssertionError as e:
            # Log the error file name to the error log file
            error_log.write(f"Error in file: pred_{scenario_id}__aid_{agent_id}__atype_{agent_type}.npz\n")
            error_log.write(f"Error message: {str(e)}\n")
            error_log.write("Stack trace:\n")
            error_log.write(traceback.format_exc())
            error_log.write("\n\n")  # Separate entries in the log file

            # Skip the current file and continue with the next iteration
            continue

        # Create a dictionary to store the results for this input file
        result_dict = {
            'probas': probas.cpu().numpy(),
            'coordinates': coordinates.cpu().numpy(),
            'covariance_matrices': covariance_matrices.cpu().numpy(),
            'loss_coeff': loss_coeff.item(),
            'xy_future_gt':xy_future_gt.cpu().numpy(),
            'target/current/speed': current_speed,
            'n_valid_timestamps': n_valid_timestamps,
            'trajectory_bucket': trajectory_bucket,
        }
        
        # Include selected_input_data items individually
        for key, value in selected_input_data.items():
            result_dict[key] = value
            
        # Move tensors to CPU before converting to NumPy arrays
        result_dict_cpu = {key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in result_dict.items()}

        #print("All keys and values in the result_dict dictionary:")
        #for key, value in result_dict.items():
            #print(f"{key}: {value}")

        # Save the results to an NPZ file
        output_file = os.path.join(output_dir, output_file_name)
        np.savez(output_file, **result_dict_cpu)

