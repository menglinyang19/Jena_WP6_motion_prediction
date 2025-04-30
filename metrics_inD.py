# create modules to compute metrics for Multipath++ algorithm
# Details of these metrics can be found out in the following three papers:
# Large Scale Interactive Motion Forecasting for Autonomous Driving : The Waymo Open Motion Dataset
# MULTIPATH++: EFFICIENT INFORMATION FUSION AND TRAJECTORY AGGREGATION FOR BEHAVIOR PREDICTION
# The Pascal Visual Object Classes (VOC) Challenge (pp. 11-12) -- for mAP metric: interpolated precision values
# Note: this is for inD dataset

import torch
from model.multipathpp import MultiPathPP
from model.data import get_dataloader, dict_to_cpu, normalize
from tqdm import tqdm
import numpy as np
import os
import re
from itertools import islice
import pandas as pd
from pandas import ExcelWriter

def euclidean_distance(point1, point2):
        # Calculate the Euclidean distance between two points.
        return np.sqrt(np.sum((point1 - point2)**2))

# calculate whether the prediction misses the ground truth for individual agent -- definitions from WOMD leaderboard
# record the highest possibility of the non-miss predictions among 6 predicted trajectories
def calculate_miss_logic(data, timestamp, gt_valid):
    miss_value = -1
    non_miss_probas = [[-1 for _ in range(6)], [0 for _ in range(6)]]
    if gt_valid[0, timestamp-1, 0] == 1:
        # scale
        initial_speed = data['target/current/speed']
        if initial_speed <= 1.4:
            scale = 0.5
        elif (initial_speed > 1.4) and (initial_speed < 11):
            scale = 0.5 + 0.5 * (initial_speed - 1.4) / (11-1.4)
        elif initial_speed >= 11:
            scale = 1

        # threshold
        if timestamp == 30:
            threshold_lat = 1
            threshold_lon = 2
        elif timestamp == 50:
            threshold_lat = 1.8
            threshold_lon = 3.6
        elif timestamp == 80:
            threshold_lat = 3.0
            threshold_lon = 6.0  

        # extract the ground truth of 3 timestamps
        gt = data['xy_future_gt']
        gt_timestamp = gt[:, timestamp-1, :]

        # extract the 6 preditions of 3 timestamps
        predict_coor = data['coordinates']
        coor_timestamp = predict_coor[0,:, timestamp-1, :]
        replicated_gt = np.tile(gt_timestamp, (6, 1))

        # calculate the distance vector
        diff_coor = replicated_gt - coor_timestamp

        dx_dis_list = []
        dy_dis_list = []

        # for 6 possible predicted trajectories
        for i in range(coor_timestamp.shape[0]):
            rotation_angle_i = np.arctan2(gt_timestamp[0, 1], gt_timestamp[0, 0]) - np.arctan2(coor_timestamp[i, 1], coor_timestamp[i, 0])

            rotation_matrix_i = np.array([
                [np.cos(rotation_angle_i), -np.sin(rotation_angle_i)],
                [np.sin(rotation_angle_i), np.cos(rotation_angle_i)]
            ])

            displacement_vector_i = np.dot(np.array([diff_coor[i,0], diff_coor[i,1]]), rotation_matrix_i)

            dx_dis_list.append(displacement_vector_i[0])
            dy_dis_list.append(displacement_vector_i[1])

        # Convert lists to NumPy arrays
        dx_dis = np.array(dx_dis_list).T  # Transpose to shape (6, 1)
        dy_dis = np.array(dy_dis_list).T  # Transpose to shape (6, 1)

        probas_orignal = data['probas']
        # Compute the log softmax along axis 1
        log_probs = np.log(np.exp(probas_orignal) / (np.sum(np.exp(probas_orignal), axis=1, keepdims=True) + 10**-6))
        # Now, log_probs contains log softmax values, and to get probabilities:
        probas = np.exp(log_probs)

        # the first row to record the miss_value
        # the second row to record the probas
        non_miss_probas = [[0 for _ in range(6)], [0 for _ in range(6)]]       
        miss_value = 0

        for i in range(6):        
            # none of the 6 predictions within the thresholds --> miss, 0; otherwise, not miss, 1
            if (np.abs(dy_dis[i]) < scale * threshold_lat) and (np.abs(dx_dis[i]) < scale * threshold_lon):
                miss_value = 1.0
                non_miss_probas[1][i] = probas[0][i]
                non_miss_probas[0][i] = miss_value
        
        # Find indices where non_miss_probas[0] is greater than 0.5
        indices_greater_than_0_5 = [i for i, value in enumerate(non_miss_probas[0]) if value > 0.5]

        # If there are indices where non_miss_probas[0] is greater than 0.5, find the index of the max value in those rows of non_miss_probas[1]
        if indices_greater_than_0_5:
            max_probas_index = indices_greater_than_0_5[np.argmax([non_miss_probas[1][i] for i in indices_greater_than_0_5])]
        else:
            # Handle the case where there are no elements in non_miss_probas[0] greater than 0.5
            max_probas_index = None 

        # Set the desired index to 1 and the rest to 0
        if not non_miss_probas:
            non_miss_probas[0][:] = [0] * len(non_miss_probas[0])
            non_miss_probas[0][max_probas_index] = 1

    return miss_value, non_miss_probas

# the distance error at timestamp t of ground truth and the closest prediction for individual agent 
def minDE(data, timestamp, gt_valid):
    min_de = float('inf')

    if gt_valid[0, timestamp-1, 0] == 1:
        # Your existing code to calculate distance_error
        distance_error_list = []

        ground_truth_trajectory = data['xy_future_gt']
        # Get the 6 predicted trajectories.
        predicted_trajectory = data['coordinates']
        # print("shape of predicted_trajectory", predicted_trajectory.shape)

        for i in range(predicted_trajectory.shape[1]):
            distance_error = 0.0
            pred_point = predicted_trajectory[0, i, timestamp-1, :]
            gt_point = ground_truth_trajectory[0, timestamp-1, :]
            # gt_point = gt_point.reshape(-1)
            # print(pred_point.shape, gt_point.shape)
            distance_error = euclidean_distance(pred_point, gt_point)

            distance_error_list.append(distance_error)

        # Convert the list to a NumPy array
        distance_error_array = np.array(distance_error_list)

        # Find the index i with the minimum distance_error
        min_error_index = np.argmin(distance_error_array)
        min_de = distance_error_array[min_error_index]    

    else: 
        min_de = -1

    return min_de

# the average distance error over all timesteps of ground truth and the closest prediction for individual agent 
def minADE(data, timestamp, gt_valid):
    min_ade = float('inf')

    if gt_valid[0, timestamp-1, 0] == 1:
        # Your existing code to calculate total_distance_error
        total_distance_error_list = []

        ground_truth_trajectory = data['xy_future_gt']
        # Get the 6 predicted trajectories.
        predicted_trajectory = data['coordinates']
        # print("shape of predicted_trajectory", predicted_trajectory.shape)

        for i in range(predicted_trajectory.shape[0]):
            total_distance_error = 0.0
            for t in range(timestamp):
                pred_point = predicted_trajectory[0, i, t, :]
                gt_point = ground_truth_trajectory[0, t, :]
                distance = euclidean_distance(pred_point, gt_point)
                total_distance_error += distance * gt_valid[0, t, 0]
                
            total_distance_error_list.append(total_distance_error)

        # Convert the list to a NumPy array
        total_distance_error_array = np.array(total_distance_error_list)

        # Find the index i with the minimum total_distance_error
        min_error_index = np.argmin(total_distance_error_array)
        min_error_value = total_distance_error_array[min_error_index]    

        min_ade = min_error_value/sum(gt_valid[0, 0:timestamp-1, 0])

    else: 
        min_ade = -1

    return min_ade    

# calculate metrics of average values for all agents
class Metrics:
    def __init__(self):
        # Initialize empty dictionaries for miss rates
        self.overall_miss_rates = {30: 0.0, 50: 0.0, 80: 0.0}
        self.agent_type_miss_rates = {30: {}, 50: {}, 80: {}}
        self.overall_min_ade = {30: 0.0, 50: 0.0, 80: 0.0}
        self.agent_type_min_ade = {30: {}, 50: {}, 80: {}}
        self.overall_min_de = {30: 0.0, 50: 0.0, 80: 0.0}
        self.agent_type_min_de = {30: {}, 50: {}, 80: {}}

    # calculate the miss rate for a certain type of road users
    def cal_miss_rate(self, missed_per_agent_timestamp):
        # Exclude values of -1 from the list
        filtered_missed = [val for val in missed_per_agent_timestamp if val >= -0.5]

        # Calculate miss rate for the filtered list
        total_agents = len(filtered_missed)
        
        if total_agents == 0:
            return 0.0

        missed_agents = sum(1 for val in filtered_missed if val < 1.0)
        miss_rate = missed_agents / total_agents
        return miss_rate

    # calculate the miss rate for all road users
    def calculate_all_miss_rates(self, agent_type_lists):
        miss_rate_results = {}
        combined_values = []

        for key, values in agent_type_lists.items():
            miss_rate_results[key] = self.cal_miss_rate(values)
            combined_values.extend(values)

        combined_result = self.cal_miss_rate(combined_values)

        return combined_result, miss_rate_results

    def mean_minDE(self, agent_type_lists):
        combined_values = []
        agent_type_min_de = {}

        for key, values in agent_type_lists.items():
            filtered_values = [val for val in values if val >= 0]

            # Calculate the mean for the filtered values
            agent_type_min_de[key] = np.mean(filtered_values)

            combined_values.extend(filtered_values)

        combined_min_de = np.mean(combined_values)
        return combined_min_de, agent_type_min_de

    def mean_minADE(self, agent_type_lists):
        combined_values = []
        agent_type_min_ade = {}

        for key, values in agent_type_lists.items():
            filtered_values = [val for val in values if val >= 0]

            # Calculate the mean for the filtered values
            agent_type_min_ade[key] = np.mean(filtered_values)

            combined_values.extend(filtered_values)

        combined_min_ade = np.mean(combined_values)
        return combined_min_ade, agent_type_min_ade

    # TODO: check this algotithm, pay attention to bucket_type_lists[bucket_i]
    def calculate_mAP(self, bucket_type_lists):
        new_bucket_type_list = {}
        for key, value in bucket_type_lists.items():
            key_name = key[0]  # Extract the single element from the tuple
            new_bucket_type_list[key_name] = value

        bucket_type_ap = {}

        mAP = 0.0
        num_buckets = 0
        # print(new_bucket_type_list)
        for bucket_key in new_bucket_type_list:
            ap = self.calculate_AP(new_bucket_type_list[bucket_key])
            bucket_type_ap[bucket_key] = ap
            if ap >= 0:
                num_buckets += 1
                mAP += ap

        mAP /= num_buckets
        return mAP, bucket_type_ap

    # for each timestamps
    def calculate_AP(self, bucket_type_lists):
        # Calculate precision and recall for the given behavior bucket, for each timestamp and each road user type. 
        # the first row non_miss_probas (value = 1.0) is marked as true positive.
        precision = []
        recall = []
        gt_num = []
        num_all_prediction = 0
        num_true_positives = 0  
        
        filtered_arrays = [arr for arr in bucket_type_lists if arr[0][0] > -1]

        if not filtered_arrays:
            ap = -1
        else: 
            list_combined = np.concatenate(filtered_arrays, axis = 1)
            
            # mark the serial numbers of agents to all probas and miss values
            num_of_agent = len(filtered_arrays)
            serial_numbers = np.array([i for i in range(num_of_agent) for _ in range(6)]).reshape(1, -1)
            list_combined = np.vstack((list_combined, serial_numbers))

            # to rank all the probas of predicted trajectories 
            sorted_indices = np.argsort(list_combined[1,:])[::-1]

            # Calculate cumulative sum of true positives and cumulative count of all predictions
            cumulative_true_positives = np.cumsum(list_combined[0, sorted_indices] > 0.5)
            cumulative_all_predictions = np.arange(1, len(sorted_indices) + 1)

            # Calculate precision
            precision = cumulative_true_positives / cumulative_all_predictions

            # Get unique agent numbers and calculate cumulative count of unique agents
            unique_agent_numbers, unique_agent_counts = np.unique(list_combined[2, sorted_indices], return_counts=True)
            cumulative_unique_agents = np.cumsum(unique_agent_counts)

            # Calculate recall
            recall = cumulative_true_positives / cumulative_unique_agents[unique_agent_numbers.searchsorted(list_combined[2, sorted_indices])]
   
            # Step 1: Sort by precision in descending order
            sorted_indices_prec = sorted(range(len(precision)), key=lambda k: precision[k], reverse=True)
            sorted_precision = [precision[i] for i in sorted_indices_prec]
            sorted_recall = [recall[i] for i in sorted_indices_prec]
                
            #sorted_indices_recall = sorted(range(len(recall)), key=lambda k: recall[k], reverse=False)
            #sorted_recall = [recall[i] for i in sorted_indices_recall]

            # Step 2: Interpolate precision values
            interpolated_precision = []
            max_precision = 0.0

            for p in sorted_precision:
                max_precision = max(max_precision, p)
                interpolated_precision.append(max_precision)

            # Step 3: Calculate AP (Area Under the Precision-Recall curve)
            recall_levels = [i / 10.0 for i in range(11)]
            ap = 0.0

            for rl in recall_levels:
                max_prec_at_recall = max([prec for prec, rec in zip(interpolated_precision, sorted_recall) if rec >= rl], default=0.0)
                ap += max_prec_at_recall

            ap /= 11.0  # Mean of maximum precision values at 11 recall levels

        return ap
    
        # Calculate the area under the precision-recall curve (AP).
    
# Specify the path to the prediction result files
file_path = "/home/meya174e/bin/inD_data/model_0802/predict_result"

# Create an instance of Metrics before the loop
metrics_calculator = Metrics()

# Dictionary to store miss rates for each agent type at different timestamps and for all agents at different timestamps
miss_rates_by_agent_type = {30: {}, 50: {}, 80: {}}
overall_miss_rates = {30: 0.0, 50: 0.0, 80: 0.0}
min_ade_by_agent_type = {30: {}, 50: {}, 80: {}}
overall_min_ade = {30: 0.0, 50: 0.0, 80: 0.0}
min_de_by_agent_type = {30: {}, 50: {}, 80: {}}
overall_min_de = {30: 0.0, 50: 0.0, 80: 0.0}
non_miss_probas_by_bucket_type = {30: {}, 50: {}, 80: {}}
AP_by_bucket_type = {30: {}, 50: {}, 80: {}}
overall_mAP = {30: 0.0, 50: 0.0, 80: 0.0}

# Define the specific timestamps
timestamps = [30, 50, 80]
behavior_buckets = ["stationary", "straight", "straight_right", "straight_left", "right_u_turn", "right_turn", "left_u_turn", "left_turn"]

# Read the file names from the generated file
file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]

#for file_name in tqdm(islice(file_names, 500), desc="Processing files"):
for file_name in tqdm(file_names, desc="Processing files"):
    # Extract relevant information from the file name using regular expression
    match = re.match(r"pred_\d+__aid_\d+__atype_(\d+)\.npz", file_name)

    if match:
        agent_type = int(match.group(1))
        # print("Agent type:", agent_type)
    else:
        # Handle the case where the filename doesn't match the expected pattern
        print(f"Error: File name '{file_name}' does not match the expected pattern.")
        continue

    # Construct the full path to the file
    full_file_path = os.path.join(file_path, file_name)

    # Read the file and perform processing (similar to the original code)
    with np.load(full_file_path) as data:
        # Create an instance of the AgentFilter class
        # n_valid_timestamps  = data['n_valid_timestamps']  # Call the method
        gt_valid = data['target/future/valid']

        agent_type_key = f"agent_type_{agent_type}"

        bucket_type = data['trajectory_bucket']
        bucket_key = tuple(bucket_type.flatten())

        # Update metrics for each agent
        for timestamp in timestamps:
            # calculate for each individual
            is_miss, non_miss_probas = calculate_miss_logic(data, timestamp, gt_valid)

            # Save the agent type-specific miss rate for the current timestamp
            if agent_type_key not in miss_rates_by_agent_type[timestamp]:
                miss_rates_by_agent_type[timestamp][agent_type_key] = []
            miss_rates_by_agent_type[timestamp][agent_type_key].append(is_miss)

            # collect maximum probas of non_missed trajectory for each agent
            if bucket_key not in non_miss_probas_by_bucket_type[timestamp]:
                non_miss_probas_by_bucket_type[timestamp][bucket_key] = []
            non_miss_probas_by_bucket_type[timestamp][bucket_key].append(non_miss_probas)

            min_de = minDE(data, timestamp, gt_valid)
            if agent_type_key not in min_de_by_agent_type[timestamp]:
                min_de_by_agent_type[timestamp][agent_type_key] = []
            min_de_by_agent_type[timestamp][agent_type_key].append(min_de) 

            min_ade = minADE(data, timestamp, gt_valid)
            if agent_type_key not in min_ade_by_agent_type[timestamp]:
                min_ade_by_agent_type[timestamp][agent_type_key] = []
            min_ade_by_agent_type[timestamp][agent_type_key].append(min_ade)  

  
for timestamp in timestamps:
    overall_miss_rates_time, agent_type_miss_rates_time = metrics_calculator.calculate_all_miss_rates(miss_rates_by_agent_type[timestamp])
    overall_miss_rates[timestamp] = overall_miss_rates_time
    miss_rates_by_agent_type[timestamp] = agent_type_miss_rates_time

    overall_min_de_time, agent_type_min_de_time = metrics_calculator.mean_minDE(min_de_by_agent_type[timestamp])
    overall_min_de[timestamp] = overall_min_de_time
    min_de_by_agent_type[timestamp] = agent_type_min_de_time

    overall_min_ade_time, agent_type_min_ade_time = metrics_calculator.mean_minADE(min_ade_by_agent_type[timestamp])
    overall_min_ade[timestamp] = overall_min_ade_time
    min_ade_by_agent_type[timestamp] = agent_type_min_ade_time

    # still need to be adjusted
    #print(non_miss_probas_by_bucket_type[timestamp])
    overall_mAP_time, bucket_type_AP_time = metrics_calculator.calculate_mAP(non_miss_probas_by_bucket_type[timestamp])
    AP_by_bucket_type[timestamp] = bucket_type_AP_time
    overall_mAP[timestamp] = overall_mAP_time
    

print(f" Overall miss rate: {overall_miss_rates}")
print(f" miss_rates_by_agent_type: {miss_rates_by_agent_type}")
print(f" overall_min_de: {overall_min_de}")
print(f" min_de_by_agent_type: {min_de_by_agent_type}")
print(f" overall_min_ade: {overall_min_ade}")
print(f" min_ade_by_agent_type: {min_ade_by_agent_type}")
print(f" overall_mAP: {overall_mAP}")
print(f" mAP_by_bucket_type: {AP_by_bucket_type}")


# Save variables to NPZ file
np.savez("/home/meya174e/bin/inD_data/model_0802/matrics_inD.npz",
         overall_miss_rates=overall_miss_rates,
         miss_rates_by_agent_type=miss_rates_by_agent_type,
         overall_min_de=overall_min_de,
         min_de_by_agent_type=min_de_by_agent_type,
         overall_min_ade=overall_min_ade,
         min_ade_by_agent_type=min_ade_by_agent_type,
         overall_mAP=overall_mAP,
         AP_by_bucket_type=AP_by_bucket_type
         )

excel_file = '/home/meya174e/bin/inD_data/model_0802/matrics_inD.xlsx'

# transform to DataFrame
df1 = pd.DataFrame.from_dict(overall_miss_rates, orient='index', columns=['Overall_miss_rate'])
df2 = pd.DataFrame(miss_rates_by_agent_type)
df3 = pd.DataFrame.from_dict(overall_min_de, orient='index', columns=['overall_min_de'])
df4 = pd.DataFrame(min_de_by_agent_type)
df5 = pd.DataFrame.from_dict(overall_min_ade, orient='index', columns=['overall_min_ade'])
df6 = pd.DataFrame(min_ade_by_agent_type)
df7 = pd.DataFrame.from_dict(overall_mAP, orient='index', columns=['overall_mAP'])
df8 = pd.DataFrame(AP_by_bucket_type)

# save DataFrame into Excel sheets:
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    # write the tables to specific locations
    df1.to_excel(writer, sheet_name='miss_rates', startrow=0, startcol=0)
    df2.to_excel(writer, sheet_name='miss_rates', startrow=len(df1) + 2, startcol=0)
    df3.to_excel(writer, sheet_name='min_de', startrow=0, startcol=0)
    df4.to_excel(writer, sheet_name='min_de', startrow=len(df3) + 2, startcol=0)
    df5.to_excel(writer, sheet_name='min_ade', startrow=0, startcol=0)
    df6.to_excel(writer, sheet_name='min_ade', startrow=len(df5) + 2, startcol=0)
    df7.to_excel(writer, sheet_name='mAP', startrow=0, startcol=0)
    df8.to_excel(writer, sheet_name='mAP', startrow=len(df7) + 2, startcol=0)