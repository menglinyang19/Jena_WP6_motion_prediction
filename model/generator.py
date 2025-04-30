import os
import cv2
import numpy as np

def image_generator(directory, batch_size):
    file_list = os.listdir(directory)
    num_files = len(file_list)
    steps_per_epoch = num_files // batch_size

    for step in range(steps_per_epoch):
        batch_files = file_list[step * batch_size: (step + 1) * batch_size]
        batch_images = []

        for file_name in batch_files:
            image = cv2.imread(os.path.join(directory, file_name))
            # Preprocess or transform the image as needed
            # e.g., resize, normalize, convert to tensors

            batch_images.append(image)

        yield np.array(batch_images)
