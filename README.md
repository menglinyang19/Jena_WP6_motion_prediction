# Motion Prediction Code (with inD Dataset)

This repository contains code for motion prediction tasks using the inD dataset.  

## 🚦 How to Use

**Important:** Before running any code, make sure to replace all occurrences of `<your_path>` in the code files with the correct path to your local data and configuration files.

**Step 1: Prepare Your Environment**  
   Use Python 3.10+ and install required packages (`requirements.txt` is provided):

**Step 2: Prepare and Preprocess the inD Dataset**

    To run the model later, you first need to convert the raw inD dataset into the required format. Follow these steps:

   1. **Preprocess raw tracking data**
      - Run `data_process.py` to extract the relevant tracking data,
      - This includes:
        - **Downsampling** the original 25Hz data to 10Hz.
        - **Converting angle values** from degrees to radians to ensure compatibility with mathematical operations.

   2. **Preprocess raw geometry data**
      - Run `geometry_process3.py` to extract and process geometry data,

   3. **Preprocess and combine tracking and geometry data**
      - Run `data_process2.py` to preprocess and reorganize the tracking data, and combine geometry data into the required format.
      - This also split all data samples into training, validation and testing sets.

**Step 3: Run Pre-rendering**
    - Run `prerender_inD.py` to pre-render the data for further model training.
    - Run the file on the training, validation, and testing sets separately. 

**Step 4: Train the model**
     - Run `train.py` with the training and validation sets. 
 
**Step 5: Assess the trained model**
     - Run `predict.py` on the testing sets and get result files of each prediction scenario.
     - Run `metrics_inD.py` on the prediction results to obtain metric values and assess the prediction accuracy.
     - Run `static_prediction_illu.py` and `dynamic_prediction_illu.py` on the selected prediction result files to create the illustration. 

  
  
