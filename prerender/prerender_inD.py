import multiprocessing
from tqdm import tqdm
import tensorflow as tf
from utils.prerender_utils import get_visualizers, create_dataset, parse_arguments, merge_and_save
from utils.utils import get_config
from utils.features_description import generate_features_description
import os
import logging
import numpy as np
logging.basicConfig(level=logging.DEBUG)
import itertools
import pdb

def main():
    # original data needed to be prerendered
    data_path = "<your_path>/validation"  # change to 'training' or 'testing' as needed 
    # where to save the prerendered data
    output_path = os.path.join('<your_path>/validation_processed') # change to 'training_processed' or 'testing_processed' as needed 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    n_jobs = 24
    n_shards = 1
    shard_id = 0
    config = "<your_path>/configs/prerender.yaml"       
    # the arguments will be passed to the script
    dataset = create_dataset(data_path, n_shards, shard_id)

    visualizers_config = get_config(config)
    visualizers = get_visualizers(visualizers_config)
    p = multiprocessing.Pool(n_jobs)  # create a group of worker processes what can execute tasks in parallel.
    # This class can provide an interface for creating and managing worker processes
    processes = []
    k = 0
     
    #for data in tqdm(dataset.as_numpy_iterator()):
    #for data_npz in tqdm(itertools.islice(dataset, 20)):
    for data_npz in tqdm(dataset):
        k += 1
        #data = tf.io.parse_single_example(data, generate_features_description())
        with data_npz as data:
            #array_to_process = data['data']

            array_to_process = {key: data[key] for key in data.files}
            #print(array_to_process["roadgraph_samples/valid"])
            #print(type(array_to_process))
            
            processes.append(
                # .apply_async: assign tasks to the worker processes
                p.apply_async(
                    merge_and_save,
                    kwds=dict(
                        visualizers=visualizers,
                        data=array_to_process,
                        output_path = output_path,
                    ),
                )
            )
    # The loop continues until all examples in the dataset have been processed

    for r in tqdm(processes):
        try:
            r.get()
        except Exception as e:
            print(f"Exception in worker process: {e}")
        #r.get()


if __name__ == "__main__":
    main()