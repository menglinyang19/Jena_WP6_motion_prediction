# Motion Prediction Code (with inD Dataset)

This repository contains code for motion prediction tasks using the inD dataset.  

## 🚦 How to Use

1. **Prepare Your Environment**  
   Use Python 3.10+ and install required packages (`requirements.txt` is provided):

2. **Step 2: Prepare and Preprocess the inD Dataset**

    To run the model later, you first need to convert the raw inD dataset into the required format. Follow these steps:

   (1). **Preprocess raw tracking data**
      - Run `data_process.py` to extract the relevant tracking data,
      - This includes:
        - **Downsampling** the original 25Hz data to 10Hz.
        - **Converting angle values** from degrees to radians to ensure compatibility with mathematical operations.

   (2). **Preprocess raw geometry data**
      - Run `geometry_process3.py` to extract and process geometry data,

`  (3). **Preprocess and combine tracking and geometry data**
      - Run `data_process2.py` to preprocess and reorganize the tracking data, and combine geometry data into the required format.
      - This also split all data samples into training, validation and testing sets.


   3. **(Optional) Visualize scenes**
      - Run `render_scenes.py` to generate trajectory or scene visualizations for inspection.
      - This step is optional but helpful for debugging or understanding the dataset.
        ```bash
        python render_scenes.py
        ```

   4. **Update file paths**
      - In all scripts, replace placeholder paths like `<your_path>/validation` with your actual local directory.
      - Example:
        ```python
        data_path = "/home/yourname/inD_data/validation"
        ```

   5. **Verify file structure**
      - Make sure all necessary intermediate files have been generated and placed in the expected directories.
  
