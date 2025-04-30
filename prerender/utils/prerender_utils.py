# add the degrees_to_radians function for data processing
import tensorflow as tf
import os
import numpy as np
from .vectorizer import MultiPathPPRenderer
from .utils import get_config, data_to_numpy
import argparse
import logging
import torch

logging.basicConfig(level=logging.DEBUG)

def get_visualizer(renderer_name, renderer_config):
    if renderer_name == "MultiPathPPRenderer":
        return MultiPathPPRenderer(renderer_config)
    raise Exception(f"Unknown visualizer {renderer_name}")

def get_visualizers(visualizers_config):
    visualizers = []
    for renderer in visualizers_config:
        visualizers.append(get_visualizer(renderer["renderer_name"], renderer["renderer_config"]))
    return visualizers

def create_dataset(datapath, n_shards, shard_id):
    files = os.listdir(datapath)
    reversed_files = sorted(files, reverse=True)
    
    dataset = []
    for file_name in reversed_files:
        data = np.load(os.path.join(datapath, file_name), allow_pickle=True)
        dataset.append(data)

    if n_shards is not None and n_shards > 1:
        dataset_size = len(dataset)
        shard_size = (dataset_size + n_shards - 1) // n_shards
        start_index = shard_id * shard_size
        end_index = min((shard_id + 1) * shard_size, dataset_size)
        dataset = dataset[start_index:end_index]
        
    return dataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save data")
    parser.add_argument("--n-jobs", type=int, default=20, required=False, help="Number of threads")
    parser.add_argument(
        "--n-shards", type=int, default=8, required=False, help="Use `1/n_shards` of full dataset")
    parser.add_argument(
        "--shard-id", type=int, default=0, required=False, help="Take shard with given id")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    return args

# how to generate names of preprocessed data files
def generate_filename(scene_data):
    scenario_id = scene_data["scenario_id"]
    agent_id = scene_data["agent_id"]
    agent_type = scene_data["target/agent_type"]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{agent_type.item()}.npz"


# apply multiple visualizers to generate multiple sets of preprocessed data, capturing different aspects of features;
# these sets of preprocessed data are then merged together

def degrees_to_radians(tensor):
    return tensor * (torch.pi / 180)

def merge_and_save(visualizers, data, output_path):
    #logging.debug("Inside merge_and_save function")
    try:
        preprocessed_dicts = [visualizer.render(data) for visualizer in visualizers]

        for scene_number in range(len(preprocessed_dicts[0])):
            scene_data = {}
            for visualizer_number in range(len(preprocessed_dicts)):
                #for visualizer_number in range(len(preprocessed_dicts)):
                scene_data.update(preprocessed_dicts[visualizer_number][scene_number])
            if scene_data:
                np.savez_compressed(os.path.join(output_path, generate_filename(scene_data)), **scene_data)
                    
    except Exception as e:
        print(f"Exception in merge_and_save: {e}")