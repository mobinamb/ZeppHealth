# GPS-RoNIN Integrated Trajectory Prediction

## Overview

This project aims to enhance trajectory prediction by integrating strong synchronized GPS data with RoNIN-LSTM model outputs. The core idea is to leverage the strengths of both systems: the global accuracy of GPS and the high-frequency, relative motion tracking of IMU-based RoNIN. This combined approach, processed through a proposed neural network, seeks to deliver more accurate and robust position estimations, particularly in challenging environments where either GPS or IMU data alone might be insufficient.

## Features

* **RoNIN-LSTM Trajectory Generation:** Utilize a pre-trained RoNIN-LSTM model to generate initial pedestrian trajectories from IMU data.
* **GPS-RoNIN Integration:** A novel network architecture that fuses synchronized GPS data with RoNIN predictions to refine position estimates.
* **Performance Evaluation:** Comprehensive evaluation of the proposed network's accuracy using metrics like Relative Trajectory Error (RTE) and Root Mean Squared Error (RMSE).
* **Trajectory Visualization:** Tools to plot and compare predicted trajectories against ground truth and synthetic GPS data for intuitive analysis.



### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mobinamb/GPSAssistedTrajectoryPrediction/
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This section guides you through generating RoNIN outputs, training your proposed network, and testing its performance.

### 1. Generating RoNIN-LSTM Model Output

This step generates the output of the RoNIN-LSTM model and saves it to the `outputs` folder. This output serves as an input to the proposed network.

```bash
python source/ronin_lstm_tcn.py test --type lstm_bi --test_list lists/list_test_unseen.txt --data_dir data/train/GT-data --out_dir outputs/resnet_lstm --model_path models/ronin_lstm/checkpoints/ronin_lstm_checkpoint.pt
```

Note that due to the limited space, I didn't add the synthetic GPS and ground truth data. They can be downlaoded locally and placed in the train and test folders of data folder

### 2. Plotting Predicted Trajectories

Gain deeper insights into the model's performance and data characteristics by visualizing the predicted trajectories. This script allows you to compare trajectories derived from IMU data, synthetic GPS data, and the ground truth for a single unseen test sample.

```bash
python utils/plotting_sample.py
```

### 3. Training the Proposed Network

Train the proposed network to integrate the strong synchronized GPS data and the predicted RoNIN data (with an offset from Ground Truth) to predict accurate positions on seen test data.

```bash
python source/proposed_network_train.py
```

**Note:** The `proposed_network_train.py` script accepts several arguments that can be specified to customize the training process, such as learning rate, batch size, number of epochs, and model saving paths. For a full list of available arguments and their descriptions, run `python source/proposed_network_train.py --help` or refer to the script's source code.

### 4. Testing the Proposed Network

Test the trained proposed network on unseen test data to determine the Relative Trajectory Error (RTE) and Root Mean Squared Error (RMSE) metrics. This script will also generate compelling plots of the resulting trajectories for visual analysis of performance, allowing for clear comparison against ground truth.

```bash
python source/proposed_network_test.py
```

## Project Structure

```
.
├── data/                       # Contains ground truth and synthetic GPS data
│   ├── train       
│   ├── test 
├── lists/                      # Lists for test and training data
├── models/                     # Pre-trained models (e.g., RoNIN-LSTM checkpoints)
├── outputs/                    # Generated trajectory outputs
├── source/
│   ├── ronin_lstm_tcn.py       # Script for RoNIN-LSTM model operations
│   ├── proposed_network_train.py # Script for training the proposed network
│   └── proposed_network_test.py  # Script for testing the proposed network
├── utils/
│   └── plotting_sample.py      # Utility for plotting trajectories
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

