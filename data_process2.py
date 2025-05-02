# This code is to further process the inD data into the format for Multipath++ model
# 1. stationary vehicles are excluded for prediction
# 2. the data is divided into training, validation and testing sets
from hmac import new
import numpy as np
import math
import xml.etree.ElementTree as ET
from collections import OrderedDict
import os
import pandas as pd
import utm
import csv
from collections import defaultdict

# determine the stationary vehicles
def get_trajectory_class(future_xy, future_speed):
    kMaxSpeedForStationary = 1.0                 # (m/s)
    kMaxDisplacementForStationary = 5.0          # (m)

    xy_delta = future_xy[:,-1] - future_xy[:,0]
    final_displacement = np.linalg.norm(xy_delta)      # euclidean distance
    max_speed = np.max(future_speed)

    if max_speed < kMaxSpeedForStationary and \
            final_displacement < kMaxDisplacementForStationary:
        return "stationary"
    return None

def get_agent_data(new_agents_data, new_meta_data, num_agents):    
    # Initialize a list to hold indices of non-stationary vehicles
    non_stationary_indices = []

    for i in range(num_agents):
        agents_data_i = new_agents_data[f"agent_id_{i}"]
        future_xy = agents_data_i[:,4:5]
        future_speed = np.sqrt(agents_data_i[:,9]**2 + agents_data_i[:,10]**2)
        trajectory_bucket = get_trajectory_class(future_xy, future_speed)

        if trajectory_bucket != "stationary":
            # Record the index of the non-stationary agent
            non_stationary_indices.append(i)

    # Ensure new_meta_data is a numpy array
    if not isinstance(new_meta_data, np.ndarray):
        new_meta_data = np.array(new_meta_data)

    non_stationary_rows = new_meta_data[non_stationary_indices]
    
    min_frame = np.min(non_stationary_rows[:,2]) + 10
    max_frame = np.max(non_stationary_rows[:,3]) - 30   # the max frame which can be selected as current frame
    dynamic_data_10Hz = {}

    #updated_num_agents = len(non_stationary_indices)

    prediction_counts = []
    non_stationary_counts = []
    one_before_set = []
    two_before_set = []
    # Every 10 frame is treated as the current moment
    for frame in range(min_frame, max_frame+1, 10):
    # for frame in range(31, max_frame+1):
        current_time = frame/10
        agents_moment = {}
        agents_id = []
        agents_type = []
        agents_width = []
        agents_length = []
        future_valid = []
        non_stationary_agents_indices = []
        for key in ["past", "current", "future"]:
            agents_moment[f"state/{key}/x"] = []
            agents_moment[f"state/{key}/y"] = []
            agents_moment[f"state/{key}/bbox_yaw"] = []
            agents_moment[f"state/{key}/velocity_x"] = []
            agents_moment[f"state/{key}/velocity_y"] = []
            agents_moment[f"state/{key}/speed"] = []
            agents_moment[f"state/{key}/valid"] = []
            agents_moment[f"state/{key}/width"] = []
            agents_moment[f"state/{key}/length"] = []

        # to check agents shown in the intersection at the current moment and the past 10 time steps (1 sec)
        for i in range(num_agents):
            if new_meta_data[i,2] + 10 <= current_time*10 and new_meta_data[i,3]-30 >= current_time*10:
                agents_data_i = new_agents_data[f"agent_id_{i}"]
                agents_id.append(new_meta_data[i,1])
                if i in non_stationary_indices:
                    non_stationary_agents_indices.append(len(agents_id)-1)

                # current and past data are all valid; future data valid:
                future_valid_i = np.ones((1,80))
                if new_meta_data[i,3] - current_time*10 >= 80:
                    num_future_valid = 80
                else:
                    num_future_valid = int(new_meta_data[i,3] - current_time*10)
                    future_valid_i[0,num_future_valid:] = 0
                future_valid.append(future_valid_i)

                current_indice = np.where(agents_data_i[:,2] == current_time*10)[0][0]

                # x, y, bbox_yaw,velocity_x, velocity_y
                for key in ["past", "current", "future"]:
                    if key == "past":
                        agents_moment[f"state/{key}/x"].append(agents_data_i[current_indice-10:current_indice,4])
                        agents_moment[f"state/{key}/y"].append(agents_data_i[current_indice-10:current_indice,5])
                        agents_moment[f"state/{key}/bbox_yaw"].append(agents_data_i[current_indice-10:current_indice,6])
                        agents_moment[f"state/{key}/velocity_x"].append(agents_data_i[current_indice-10:current_indice,9])
                        agents_moment[f"state/{key}/velocity_y"].append(agents_data_i[current_indice-10:current_indice,10])
                        speed_past = np.sqrt(agents_data_i[current_indice-10:current_indice,9]**2 + agents_data_i[current_indice-10:current_indice,10]**2)
                        agents_moment[f"state/{key}/speed"].append(speed_past)
                        agents_moment[f"state/{key}/valid"].append(np.ones((1,10)))
                        agents_moment[f"state/{key}/width"].append(agents_data_i[current_indice-10:current_indice,7])
                        agents_moment[f"state/{key}/length"].append(agents_data_i[current_indice-10:current_indice,8])
                    elif key == "current":
                        agents_moment[f"state/{key}/x"].append(agents_data_i[current_indice,4])
                        agents_moment[f"state/{key}/y"].append(agents_data_i[current_indice,5])
                        agents_moment[f"state/{key}/bbox_yaw"].append(agents_data_i[current_indice,6])
                        agents_moment[f"state/{key}/velocity_x"].append(agents_data_i[current_indice,9])
                        agents_moment[f"state/{key}/velocity_y"].append(agents_data_i[current_indice,10])
                        agents_moment[f"state/{key}/speed"].append(agents_data_i[current_indice,11])
                        agents_moment[f"state/{key}/valid"].append(1)
                    else:
                        if num_future_valid == 80:
                            agents_moment[f"state/{key}/x"].append(agents_data_i[current_indice+1:current_indice+num_future_valid+1,4].reshape(1,-1))
                            agents_moment[f"state/{key}/y"].append(agents_data_i[current_indice+1:current_indice+num_future_valid+1,5].reshape(1,-1))
                            agents_moment[f"state/{key}/bbox_yaw"].append(agents_data_i[current_indice+1:current_indice+num_future_valid+1,6].reshape(1,-1))
                            agents_moment[f"state/{key}/velocity_x"].append(agents_data_i[current_indice+1:current_indice+num_future_valid+1,9].reshape(1,-1))
                            agents_moment[f"state/{key}/velocity_y"].append(agents_data_i[current_indice+1:current_indice+num_future_valid+1,10].reshape(1,-1))
                            speed_future = np.sqrt(agents_data_i[current_indice+1:current_indice+num_future_valid+1,9]**2 + agents_data_i[current_indice+1:current_indice+num_future_valid+1,10]**2)
                            agents_moment[f"state/{key}/speed"].append(speed_future.reshape(1,-1))
                            agents_moment[f"state/{key}/valid"].append(future_valid_i)
                            agents_moment[f"state/{key}/width"].append(agents_data_i[current_indice+1:current_indice+num_future_valid+1,7].reshape(1,-1))
                            agents_moment[f"state/{key}/length"].append(agents_data_i[current_indice+1:current_indice+num_future_valid+1,8].reshape(1,-1))
                        else: 
                            padding_array = -1*np.ones((1,80-num_future_valid))
                            future_x = np.concatenate((agents_data_i[current_indice+1:current_indice+num_future_valid+1,4].reshape(1,-1), padding_array), axis=1)
                            agents_moment[f"state/{key}/x"].append(future_x)
                            future_y = np.concatenate((agents_data_i[current_indice+1:current_indice+num_future_valid+1,5].reshape(1,-1),
                                                        padding_array), axis=1)
                            agents_moment[f"state/{key}/y"].append(future_y)
                            future_yaw = np.concatenate((agents_data_i[current_indice+1:current_indice+num_future_valid+1,6].reshape(1,-1),
                                                        padding_array), axis=1)
                            agents_moment[f"state/{key}/bbox_yaw"].append(future_yaw)
                            future_xvel = np.concatenate((agents_data_i[current_indice+1:current_indice+num_future_valid+1,9].reshape(1,-1),
                                                        padding_array), axis=1)
                            agents_moment[f"state/{key}/velocity_x"].append(future_xvel)
                            future_yvel = np.concatenate((agents_data_i[current_indice+1:current_indice+num_future_valid+1,10].reshape(1,-1),
                                                        padding_array), axis=1)
                            agents_moment[f"state/{key}/velocity_y"].append(future_yvel)
                            speed_future = np.sqrt(agents_data_i[current_indice+1:current_indice+num_future_valid+1,9]**2 + agents_data_i[current_indice+1:current_indice+num_future_valid+1,10]**2)
                            speed_future_all = np.concatenate((speed_future.reshape(1,-1), padding_array), axis=1)
                            agents_moment[f"state/{key}/speed"].append(speed_future_all)
                            agents_moment[f"state/{key}/valid"].append(future_valid_i)
                            future_width = np.concatenate((agents_data_i[current_indice+1:current_indice+num_future_valid+1,7].reshape(1,-1),
                                                        padding_array), axis=1)
                            agents_moment[f"state/{key}/width"].append(future_width)
                            future_length = np.concatenate((agents_data_i[current_indice+1:current_indice+num_future_valid+1,8].reshape(1,-1),
                                                        padding_array), axis=1)
                            agents_moment[f"state/{key}/length"].append(future_length)

                if new_meta_data[i,7] == "car" or new_meta_data[i,7] == "truck_bus":
                    agents_type.append(1) 
                    agents_width.append(new_meta_data[i,5])
                    agents_length.append(new_meta_data[i,6])
                elif new_meta_data[i,7] == "pedestrian": 
                    agents_type.append(2)
                    # get the distribution from SUMO or Waymo datasets???
                    agents_width.append(0.5)
                    agents_length.append(0.5)
                elif new_meta_data[i,7] == "bicycle":
                    agents_type.append(3)
                    agents_width.append(0.5)
                    agents_length.append(1.5)
            
        num_agents_frame = len(agents_id)
        # the number of non-stationary agents at this moment
        num_non_stationary_agents_frame = len(non_stationary_agents_indices)

        agents_id = np.array(agents_id)
        agents_type = np.array(agents_type)
        agents_width = np.array(agents_width)
        agents_length = np.array(agents_length)
        
        for key in ["past","current"]:
            agents_moment[f"state/{key}/x"] = np.array(agents_moment[f"state/{key}/x"])
            agents_moment[f"state/{key}/y"] = np.array(agents_moment[f"state/{key}/y"])
            agents_moment[f"state/{key}/bbox_yaw"] = np.array(agents_moment[f"state/{key}/bbox_yaw"])
            agents_moment[f"state/{key}/velocity_x"] = np.array(agents_moment[f"state/{key}/velocity_x"])
            agents_moment[f"state/{key}/velocity_y"] = np.array(agents_moment[f"state/{key}/velocity_y"])
            agents_moment[f"state/{key}/speed"] = np.array(agents_moment[f"state/{key}/speed"])
            agents_moment[f"state/{key}/valid"] = np.vstack(agents_moment[f"state/{key}/valid"])
            agents_moment[f"state/{key}/width"] = np.array(agents_moment[f"state/{key}/width"])
            agents_moment[f"state/{key}/length"] = np.array(agents_moment[f"state/{key}/length"])
        
        for key_attr in ["x","y","bbox_yaw","velocity_x","velocity_y","speed","valid"]:
            agents_moment[f"state/current/{key_attr}"] = agents_moment[f"state/current/{key_attr}"].reshape(-1,1)
        
        # adjust the shape of future data to fit the requirements of Multipath++
        for key in ["future"]:
            agents_moment[f"state/{key}/x"] = np.vstack(agents_moment[f"state/{key}/x"])
            agents_moment[f"state/{key}/y"] = np.vstack(agents_moment[f"state/{key}/y"])
            agents_moment[f"state/{key}/bbox_yaw"] = np.vstack(agents_moment[f"state/{key}/bbox_yaw"])
            agents_moment[f"state/{key}/velocity_x"] = np.vstack(agents_moment[f"state/{key}/velocity_x"])
            agents_moment[f"state/{key}/velocity_y"] = np.vstack(agents_moment[f"state/{key}/velocity_y"])
            agents_moment[f"state/{key}/speed"] = np.vstack(agents_moment[f"state/{key}/speed"])
            agents_moment[f"state/{key}/valid"] = np.vstack(agents_moment[f"state/{key}/valid"])
            agents_moment[f"state/{key}/width"] = np.vstack(agents_moment[f"state/{key}/width"])
            agents_moment[f"state/{key}/length"] = np.vstack(agents_moment[f"state/{key}/length"])
        
        ## how to determine agents as data samples
        
        # randomly choose 20% agents for prediction
        num_prediction = int(np.ceil(num_non_stationary_agents_frame * 0.5))        # how many agents will be predicted

        if num_prediction > 0:
            non_stationary_counts.append(num_non_stationary_agents_frame)
            prediction_indicator = np.zeros(num_agents_frame)

            # exclude predicted agents of one previous frames
            # we don't want to predict the same object for two consecutive time frames
            excluded_set = np.unique(np.concatenate((one_before_set, two_before_set)))

            agent_id_to_index = {agent_id: idx for idx, agent_id in enumerate(agents_id)}

            # Handle the case where there might be IDs in excluded_set that are not in agent_id_to_index
            excluded_indices = []
            for agent_id in excluded_set:
                if agent_id in agent_id_to_index:
                    excluded_indices.append(agent_id_to_index[agent_id])

            excluded_indices = np.array(excluded_indices)

            eligible_agent_indices = np.setdiff1d(non_stationary_agents_indices, excluded_indices)

            if num_prediction >= len(eligible_agent_indices):
                num_prediction = len(eligible_agent_indices)

            if num_prediction > 0:
                # Randomly select positions among non-stationary agents
                non_stationary_positions = np.random.choice(eligible_agent_indices, num_prediction, replace=False)
    
                prediction_counts.append(num_prediction)
                prediction_indicator[non_stationary_positions] = 1

                agents_moment.update ({
                    'current_time': int(current_time*1000),
                    'state/id': agents_id,
                    'state/type': agents_type,
                    'state/is_sdc': np.zeros((num_agents_frame,)),
                    'state/tracks_to_predict': prediction_indicator,
                    'state/current/width': agents_width.reshape(-1,1),
                    'state/current/length': agents_length.reshape(-1,1)
                })

                # The list under each time frame can be treated as one data record
                time_key = f"time_{current_time}"
                dynamic_data_10Hz[time_key] = agents_moment

            two_before_set = []
            two_before_set = one_before_set
            one_before_set = []
            if num_prediction > 0:
                one_before_set = agents_id[non_stationary_positions]

    # After all iterations, calculate the average value
    average_prediction_counts = sum(prediction_counts) / len(prediction_counts)
    average_non_stationary_counts = sum(non_stationary_counts) / len(non_stationary_counts)

    return dynamic_data_10Hz, average_prediction_counts, average_non_stationary_counts, sum(prediction_counts)

agent_data_dictionary = '<your_path>/inD_data/original_data/'
location_dictionary = '<your_path>/inD_data/lanelets/'
# processed data from data_process.py
newdata_dictionary = '<your_path>/inD_data/10Hz_data/'

# processed geometry data from geometry_process3.py
geometry_filename = 'geometry_info6.npz'
geometry_filepath= os.path.join(location_dictionary, geometry_filename)
geometry_data = np.load(geometry_filepath, allow_pickle=True)
node_info = dict(geometry_data)

total_samples = 0
training_prob = 0.7
validation_prob = 0.15
testing_prob = 0.15

output_filename = 'agents_num.txt'
output_filepath= os.path.join(newdata_dictionary, output_filename)
with open(output_filepath, 'w') as file:
    data_sample = []
    # This loop processes 33 tracks to calculate and log the average number of non-stationary agents
    # and the average number of agents selected for prediction in each track.
    for i in range(0, 33):
        meta_filename = f"{i:02d}_tracksMeta.csv"
        meta_filepath = os.path.join(agent_data_dictionary, meta_filename)
        df = pd.read_csv(meta_filepath)
        headers1 = df.columns.tolist()  # Extract headers directly from DataFrame
        meta_data = df.values
        num_agents = meta_data.shape[0]

        agents_filename = f"{i:02d}_tracks.csv"
        agents_filepath = os.path.join(agent_data_dictionary, agents_filename)
        df = pd.read_csv(agents_filepath)
        headers2 = df.columns[:13].tolist()  # Extract headers directly from DataFrame
        agents_data = df.values

        # read the 10Hz data
        newdata_filename1 = f"{i:02d}_tracksMeta_10Hz.csv"
        newdata_filepath1 = os.path.join(newdata_dictionary, newdata_filename1)
        new_meta_data = pd.read_csv(newdata_filepath1, header=None, skiprows=1)
        newdata_filename2 = f"{i:02d}_tracks_10Hz.csv"
        newdata_filepath2 = os.path.join(newdata_dictionary, newdata_filename2)
        concatenated_agents_data = pd.read_csv(newdata_filepath2, skiprows=1)
        newdata_filename3 = f"{i:02d}_agents_10Hz.npz"
        newdata_filepath3 = os.path.join(newdata_dictionary, newdata_filename3)
        loaded_data = np.load(newdata_filepath3)
        new_agents_data = dict(loaded_data)

        recording_filename = f"{i:02d}_recordingMeta.csv"
        recording_filepath = os.path.join(agent_data_dictionary, recording_filename)
        df = pd.read_csv(recording_filepath)
        recording_data = df.values

        # get agents' attributes and trajectories
        dynamic_data_10Hz, average_prediction_counts, average_non_stationary_counts,total_predition = get_agent_data(new_agents_data, new_meta_data, num_agents)
        data_sample.append(total_predition)
        
        # Write the statistics for each track to the output file
        file.write(f"track{i}, non_stationary: {average_non_stationary_counts}, prediction: {average_prediction_counts}\n")

        if dynamic_data_10Hz is not None:
            # get the location number and the corresponding geometry data 
            location_id = recording_data[0,1]
            key_location = f"Location_{location_id}"
            node_info_k = node_info[key_location].item()
            #print(type(node_info_k))

            # combine the agents data and geometry data and save them as data sample files  
            for data_list in dynamic_data_10Hz:
                combined_data = {}
                agents_data_k = dynamic_data_10Hz[data_list]
                combined_data.update(agents_data_k)
                combined_data.update(node_info_k)
                current_time = combined_data['current_time']
                time_str = f"{current_time:07d}"
                location_str = f"{location_id}"
                track_str = f"{i:02d}"
                scenario_id = time_str + location_str + track_str
                combined_data['scenario/id'] = scenario_id
                del combined_data['current_time']

                #if location_id == 3:
                #    folder = 'testing' 
                #else:
                # Generate a random number between 0 and 1
                random_number = np.random.rand()

                # Determine the folder based on the random number
                if random_number < training_prob:
                    folder = 'training'
                elif random_number >= training_prob and random_number < training_prob + validation_prob:
                    folder = 'validation' 
                else: 
                    folder = 'testing'

                # Generate the file name with a seven-digit number
                file_name = f"sample_{total_samples:07d}.npz"

                # Create the folder if it doesn't exist
                folder_path = os.path.join('/home/meya174e/bin/inD_data/model_1011/', folder)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # Save the array as a .npz file in the corresponding folder
                file_path = os.path.join(folder_path, file_name)
                np.savez(file_path, **combined_data)
                # Increment the counter
                total_samples += 1

        print("done, {:02d}".format(i))
    print(sum(data_sample))
