# UWave_gesture_recognition-

# Dataset Overview

I used the dataset from [uWave: Accelerometer-based personalized gesture recognition and its applications](https://www.yecl.org/publications/liu09percom.pdf) by Jiayang Liu et al. This dataset focuses on accelerometer-based personalized gesture recognition and consists of gesture samples recorded under controlled conditions.

### Dataset Details

- **Source**: [Dataset Link](https://www.yecl.org/publications/liu09percom.pdf)
- **Paper Reference**: Jiayang Liu et al., "uWave: Accelerometer-based personalized gesture recognition and its applications"

### Dataset Structure

Unpacking the dataset reveals several `.rar` files with the following organization:

1. **Naming Convention**:
   - Each `.rar` file corresponds to gesture samples collected from one user on a specific day.
   - Files are named as `U$userIndex($dayIndex).rar`, where:
     - `$userIndex`: Participant index (1 to 8)
     - `$dayIndex`: Day index (1 to 7)

2. **Contents of `.rar` Files**:
   - Each `.rar` file contains `.txt` files representing time-series acceleration data of gestures.
   - Header Section: Contains metadata about the dataset, including:
   - @relation: Name of the dataset.
   - @attribute: List of attributes (features), with their data types.
   
  - Data Section: Starts with the @data line, followed by rows of data entries that match the defined attributes.

   - Each `.txt` file contains time-series data of acceleration values:
     - **Column 1**: x-axis acceleration
     - **Column 2**: y-axis acceleration
     - **Column 3**: z-axis acceleration
   - **Units**: Acceleration is measured in **G** (acceleration due to gravity).

### Gestures

The dataset includes 8 distinct gestures as shown in the reference material. In the gesture diagrams:
- **Dot**: Indicates the start of a gesture
- **Arrow**: Indicates the end of a gesture
Take look into this [EDA](https://github.com/MARESH001/UWave_gesture_recognition-/blob/main/notebook/data/EDA.ipynb)

Below is an example visualization of the gestures for reference.

![Gesture Representation](gesture.png)


For additional details, refer to the [paper](https://www.yecl.org/publications/liu09percom.pdf).

## Problems Faced with the Dataset

- **Missing Values**: The dataset contained missing values, which caused uncertainty in predictions and required careful handling during preprocessing.
- **Non-Numeric Data**: Some parts of the dataset were not completely numeric, making preprocessing a necessary step to ensure compatibility with machine learning algorithms.

## Modeling Approach

I utilized **PyCaret**, one of my favorite AutoML libraries, for model training and evaluation. Among the various models tested, the **Extra Trees Classifier** and the **Random Forest Classifier** emerged as the best-performing models for this dataset, achieving F1 scores of over **88%**.

### Key Insights:

- The combination of robust preprocessing and the use of AutoML significantly streamlined the modeling process.
- Both classifiers were effective in handling the complexities of the dataset, making them ideal choices for this task.
## How to run this files
1. Run follwing command
```bash
python app.py
```
2. Test using json file will local endpoint/url in postman
