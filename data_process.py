# This code is to primarily process the original data of inD Dataset, to fit the Multipath++ model
#   1. 25 hz --> 10 Hz; 
#   2. degree --> radian
from hmac import new
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict
import os
import pandas as pd
from collections import defaultdict
import warnings
import torch

def degrees_to_radians(tensor):
    return tensor * (torch.pi / 180)

def reorganize_data_10Hz(origin_meta_data, origin_agents_data, num_agents):
    # extraction and interpolation to get data of 10Hz
    # origin: 25Hz
    initial_frame = np.ceil(origin_meta_data[:,2]*0.04/0.1).reshape(-1,1)   # under 10Hz
    final_frame = np.floor(origin_meta_data[:,3]*0.04/0.1).reshape(-1,1)    # under 10Hz
    new_meta_data = origin_meta_data[:,0:13]
    new_meta_data[:,2:4] = np.concatenate([initial_frame, final_frame], axis = 1)
    # print(np.concatenate([initial_frame, final_frame], axis = 1))

    new_agents_data = {}
    all_agents_data = []

    for k in range(num_agents):
        num_frame_k = final_frame[k,0] - initial_frame[k,0] + 1   # how many frame that agent k has under 10Hz
        agents_data_k = np.zeros((num_frame_k, 13))
        # the odd rows are the 5*n-th rows of the original agents_data
        for i in range(num_frame_k):
            origin_frame = int(np.floor((i+initial_frame[k,0]) * 0.1 / 0.04))
            condition1 = origin_agents_data[:, 2] == origin_frame  # the starting time of recording trajectories
            condition2 = origin_agents_data[:, 1] == k   # the subject agent
            po1 = np.where(condition1 & condition2)[0][0]
            if i%2 == 0:
                agents_data_k[i,:] = origin_agents_data[po1, 0:13]
            else:
                # interpolation
                agents_data_k[i,0:2] = origin_agents_data[po1, 0:2]
                agents_data_k[i,7:9] = origin_agents_data[po1, 7:9]
                if po1+1 < len(origin_agents_data):
                    agents_data_k[i,4:7] = (origin_agents_data[po1, 4:7] + origin_agents_data[po1+1, 4:7])/2
                    agents_data_k[i,9:13] = (origin_agents_data[po1, 9:13] + origin_agents_data[po1+1, 9:13])/2
                else:
                    agents_data_k[i,4:7] = origin_agents_data[po1, 4:7] 
                    agents_data_k[i,9:13] = origin_agents_data[po1, 9:13] 

            # Convert degree to radians for agents_data_k[i,6]
            agents_data_k[i, 6] = degrees_to_radians(torch.tensor(agents_data_k[i, 6])).item()

            agents_data_k[i,2] = i+initial_frame[k,0]


        agents_data_k[:,3] = np.arange(num_frame_k)

        key = f"agent_id_{k}"
        new_agents_data[key] = agents_data_k
        all_agents_data.append(agents_data_k)

    concatenated_array = np.vstack(all_agents_data)
    
    return concatenated_array, new_agents_data, new_meta_data

# original agent tracking data in csv format
agent_data_dictionary = '<your_path>/inD_data/original_data/'
# original lanelet data in xml format
location_dictionary = '<your_path>/inD_data/lanelets/'
# where to save the primarily processed agent tracking data
output_dictionary = '<your_path>/inD_data/10Hz_data/'

# Set up warning to be triggered as an exception
warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)

# totol 33 tracking data collection scenarios
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

    # save the 10Hz data
    concatenated_agents_data, new_agents_data, new_meta_data = reorganize_data_10Hz(meta_data, agents_data, num_agents)
   
    output_filename1 = f"{i:02d}_tracksMeta_10Hz.csv"
    output_filepath1= os.path.join(output_dictionary, output_filename1)
    pd.DataFrame(new_meta_data, columns=headers1).to_csv(output_filepath1, index=False)
    output_filename2 = f"{i:02d}_tracks_10Hz.csv"
    output_filepath2= os.path.join(output_dictionary, output_filename2)
    pd.DataFrame(concatenated_agents_data, columns=headers2).to_csv(output_filepath2, index=False)
    output_filename3 = f"{i:02d}_agents_10Hz.npz"
    output_filepath3 = os.path.join(output_dictionary, output_filename3)    
    np.savez(output_filepath3, **new_agents_data)

    recording_filename = f"{i:02d}_recordingMeta.csv"
    recording_filepath = os.path.join(agent_data_dictionary, recording_filename)
    df = pd.read_csv(recording_filepath)
    recording_data = df.values

    print("done, {:02d}".format(i))
