import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pyproj import Transformer
import os.path as osp
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse 

# --- PyTorch Model for Temporal Fusion ---
class TemporalFusionModel(nn.Module):
    """
    A temporal model to fuse synchronized GPS data (position, HDOP, validity)
    and RoNIN velocity predictions to estimate a more accurate position.
    It learns to weigh inputs based on GPS quality and correct RoNIN drift.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_prob=0.1):
        super(TemporalFusionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU processes the combined features over time
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob)
        
        # Fusion head outputs the fused 2D position (x, y)
        # Input to fusion head is hidden_dim * 2 due to bidirectional GRU
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # Output: fused x, y coordinates
        )

    def forward(self, x):
        # x will contain [gps_x, gps_y, hdop, valid_gps_flag, ronin_vx, ronin_vy]
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        gru_out, _ = self.gru(x, h0) # gru_out shape: (batch_size, sequence_length, hidden_dim * 2)
        
        fused_positions = self.fusion_head(gru_out) # fused_positions shape: (batch_size, sequence_length, output_dim)
        return fused_positions

# --- Dataset for Training the Temporal Fusion Model ---
class FusionDataset(Dataset):
    def __init__(self, all_model_inputs, all_gt_positions, sequence_length):
        self.sequences = []
        self.gt_targets = []

        for i in range(len(all_model_inputs)):
            full_model_input = all_model_inputs[i]
            full_gt_position = all_gt_positions[i]
            num_frames = full_model_input.shape[0]

            stride = sequence_length // 4
            if stride == 0: stride = 1 
            
            for j in range(0, num_frames - sequence_length + 1, stride):
                end_idx = j + sequence_length
                self.sequences.append(torch.tensor(full_model_input[j:end_idx], dtype=torch.float32))
                self.gt_targets.append(torch.tensor(full_gt_position[j:end_idx], dtype=torch.float32))
        
        print(f"FusionDataset created with {len(self.sequences)} sequences for training.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.gt_targets[idx]

# --- DataProcessor Class ---
class DataProcessor:
    def __init__(self, ronin_output_path, gps_data_path, imu_sampling_rate=200):
        self.ronin_output_path = ronin_output_path
        self.gps_data_path = gps_data_path
        self.imu_sampling_rate = imu_sampling_rate
        self.transformer = None # pyproj transformer, initialized on first use
        self.gps_is_latlon = False # Default to False, updated in load_gps_data

    def load_and_extract_ronin_data(self):
        ronin_data = np.load(self.ronin_output_path)
        ronin_vel_pred = ronin_data[:, 2:4] # Velocity (vx, vy)
        gt_pos = ronin_data[:, 4:6]         # Ground Truth Position (x, y)
        gt_vel = ronin_data[:, 6:8]         # Ground Truth Velocity (vx, vy)
        return ronin_vel_pred, gt_pos, gt_vel

    def load_gps_data(self):
        df_gps = pd.read_csv(self.gps_data_path)
        
        # Check for local Cartesian columns first
        required_local_cols = ['time_s', 'x_m', 'y_m', 'HDOP']
        if all(col in df_gps.columns for col in required_local_cols):
            self.gps_is_latlon = False 
            return df_gps
        
    def generate_master_timestamps(self, num_frames):
        return np.arange(num_frames) / self.imu_sampling_rate

    def _calculate_hdop_validity_flag(self, hdop_array, max_hdop=6.0):
        # This flag is 1.0 if considered 'valid' for GPS reliance, 0.0 otherwise
        return (hdop_array <= max_hdop).astype(float)

    def synchronize_data(self, df_gps, master_timestamps, ronin_vel_pred, gt_pos):
        # Determine appropriate columns for interpolation based on whether GPS is lat/lon or local
        
        gps_time_col = 'time_s'
        gps_lat_col = 'y_m'
        gps_lon_col = 'x_m'

        f_lat_or_y = interp1d(df_gps[gps_time_col], df_gps[gps_lat_col], kind='linear', fill_value="extrapolate")
        f_lon_or_x = interp1d(df_gps[gps_time_col], df_gps[gps_lon_col], kind='linear', fill_value="extrapolate")
        f_hdop = interp1d(df_gps[gps_time_col], df_gps['HDOP'], kind='linear', fill_value="extrapolate")

        synced_gps_y_local = f_lat_or_y(master_timestamps)
        synced_gps_x_local = f_lon_or_x(master_timestamps)
        synced_gps_hdop = f_hdop(master_timestamps)

        # Calculate the validity flag based on HDOP 
        HDOP_VALID_THRESHOLD = 5.0 # This threshold can be an argument too if you want to experiment
        valid_gps_flag = self._calculate_hdop_validity_flag(synced_gps_hdop, HDOP_VALID_THRESHOLD)
        
        # Also consider timestamps outside the GPS data range as invalid for GPS signal
        valid_gps_flag[master_timestamps < df_gps[gps_time_col].min()] = 0.0
        valid_gps_flag[master_timestamps > df_gps[gps_time_col].max()] = 0.0


        plt.plot(synced_gps_x_local, synced_gps_y_local, label='Synchronized GPS', color='blue', alpha=0.5)
        plt.plot(df_gps[gps_lat_col],df_gps[gps_lon_col], color='red', alpha=0.5)
        plt.show()
        # Create the input feature vector for the TemporalFusionModel
        # [gps_x_local, gps_y_local, hdop, valid_gps_flag, ronin_vx, ronin_vy]
        model_input_features = np.stack([
            synced_gps_x_local,
            synced_gps_y_local,
            synced_gps_hdop,
            valid_gps_flag,
            ronin_vel_pred[:, 0], # RoNIN vx
            ronin_vel_pred[:, 1]  # RoNIN vy
        ], axis=-1)

        # Ground truth for the model's output is the RoNIN GT Position
        # This assumes GT_pos is already in the same local Cartesian system as the synchronized GPS data
        gt_positions_for_model = gt_pos

        return model_input_features, gt_positions_for_model

    def convert_latlon_to_local_cartesian(self, lat_array, lon_array):
        first_valid_idx_arr = np.where(~np.isnan(lat_array))
        if len(first_valid_idx_arr[0]) == 0:
            print("Warning: No valid lat/lon points found for Cartesian conversion.")
            return np.full_like(lat_array, np.nan), np.full_like(lon_array, np.nan)

        first_valid_idx = first_valid_idx_arr[0][0]
        ref_lat = lat_array[first_valid_idx]
        ref_lon = lon_array[first_valid_idx]

        if self.transformer is None:
            utm_zone = int((ref_lon + 180) / 6) + 1
            self.transformer = Transformer.from_crs("epsg:4326", f"epsg:326{utm_zone}", always_xy=True)

        # pyproj transformer expects (longitude, latitude) when always_xy=True
        x_utm, y_utm = self.transformer.transform(lon_array, lat_array) 
        ref_x_utm, ref_y_utm = self.transformer.transform(ref_lon, ref_lat)

        local_x = x_utm - ref_x_utm
        local_y = y_utm - ref_y_utm
        return local_x, local_y

# --- Normalization Manager Class ---
class NormalizationManager:
    def __init__(self):
        # Input normalization stats
        self.input_mean = None
        self.input_std = None
        # Output (Ground Truth) normalization stats
        self.output_mean = None
        self.output_std = None

    def fit(self, all_model_inputs, all_gt_positions): # <--- IMPORTANT: Added all_gt_positions here
        """
        Calculates mean and std dev for both input features and GT positions
        from training data.
        """
        # Fit for inputs
        stacked_inputs = np.vstack(all_model_inputs)
        self.input_mean = np.mean(stacked_inputs, axis=0)
        self.input_std = np.std(stacked_inputs, axis=0)
        # Add a small epsilon to prevent division by zero for features with zero std dev
        self.input_std[self.input_std == 0] = 1e-6 
        
        # Fit for outputs (GT positions)
        stacked_gt = np.vstack(all_gt_positions) # <--- IMPORTANT: Use all_gt_positions here
        self.output_mean = np.mean(stacked_gt, axis=0)
        self.output_std = np.std(stacked_gt, axis=0)
        self.output_std[self.output_std == 0] = 1e-6 
        
        print(f"\n--- Normalization Statistics Calculated ---")
        print(f"Input Feature Means: {self.input_mean}")
        print(f"Input Feature Std Devs: {self.input_std}")
        print(f"Output (GT) Position Means: {self.output_mean}") # <--- Added print
        print(f"Output (GT) Position Std Devs: {self.output_std}") # <--- Added print
        print("------------------------------------------")

    def transform_inputs(self, model_input_array):
        """Applies normalization to a given input array."""
        if self.input_mean is None or self.input_std is None:
            raise ValueError("Input normalization stats not fitted/loaded yet. Call .fit() or .load_stats() first.")
        
        return (model_input_array - self.input_mean) / self.input_std

    def transform_outputs(self, gt_position_array): # <--- New method for normalizing GT
        """Applies normalization to a given GT position array."""
        if self.output_mean is None or self.output_std is None:
            raise ValueError("Output normalization stats not fitted/loaded yet. Call .fit() or .load_stats() first.")
        return (gt_position_array - self.output_mean) / self.output_std

    def denormalize_outputs(self, normalized_output_array):
        """Denormalizes a predicted output array back to its original scale."""
        if self.output_mean is None or self.output_std is None:
            raise ValueError("Output normalization stats not fitted/loaded yet. Call .fit() or .load_stats() first.")
        return (normalized_output_array * self.output_std) + self.output_mean

    def save_stats(self, filepath):
        """Saves all normalization statistics to a .npz file."""
        np.savez(filepath, 
                 input_mean=self.input_mean, 
                 input_std=self.input_std,
                 output_mean=self.output_mean, # <--- IMPORTANT: Now saving output stats
                 output_std=self.output_std)   # <--- IMPORTANT: Now saving output stats
        print(f"Normalization statistics saved to: {filepath}")

    def load_stats(self, filepath):
        """Loads all normalization statistics from a .npz file."""
        if not osp.exists(filepath):
            raise FileNotFoundError(f"Normalization stats file not found at: {filepath}")
        
        stats = np.load(filepath)
        self.input_mean = stats['input_mean']
        self.input_std = stats['input_std']
        
        # IMPORTANT: Check if keys exist, for backward compatibility and robustness
        if 'output_mean' in stats and 'output_std' in stats:
            self.output_mean = stats['output_mean']
            self.output_std = stats['output_std']
        else:
            # If the file is old and doesn't have these, raise a specific error
            raise ValueError(f"Normalization stats file '{filepath}' does not contain 'output_mean' or 'output_std'. "
                             "You likely need to retrain your model with the updated NormalizationManager to generate a new stats file.")
            
        print(f"Normalization statistics loaded from: {filepath}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train the Temporal Fusion Model for GPS/RoNIN fusion.")
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--sequence_length', type=int, default=200,
                        help='Length of each sequence fed to the GRU.')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='Hidden dimension of the GRU.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GRU layers.')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='Dropout probability for GRU layers.')
    parser.add_argument('--imu_sampling_rate', type=int, default=200,
                        help='IMU/RoNIN sampling rate in Hz.')
    parser.add_argument('--data_dir_gt', type=str, default='./data/train/GT-data',
                        help='Path to the directory containing GT data (for RoNIN outputs).')
    parser.add_argument('--data_dir_gps', type=str, default='./data/train/synthetic-GPS/seen_subjects_test_set/',
                        help='Path to the directory containing synthetic GPS data.')
    parser.add_argument('--ronin_outputs_base_dir', type=str, default='./outputs/resnet_lstm/',
                        help='Base directory for RoNIN model outputs.')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models',
                        help='Directory to save the trained model.')
    parser.add_argument('--figure_save_dir', type=str, default='./figures',
                        help='Directory to save training plots.')
    parser.add_argument('--model_name', type=str, default='temporal_fusion_model.pth',
                        help='Filename for the saved model.')
    parser.add_argument('--plot_loss', action='store_true',
                        help='Whether to plot and save the training loss curve.')



    args = parser.parse_args()

    # --- Configuration from Args ---
    INPUT_DIM = 6 # gps_x, gps_y, hdop, valid_gps_flag, ronin_vx, ronin_vy
    OUTPUT_DIM = 2 # fused_x, fused_y (position)
    
    # Use args values
    HIDDEN_DIM = args.hidden_dim
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    SEQUENCE_LENGTH = args.sequence_length
    NUM_LAYERS = args.num_layers
    DROPOUT_PROB = args.dropout_prob
    IMU_SAMPLING_RATE = args.imu_sampling_rate

    # Paths from args
    DATA_DIR_GT = args.data_dir_gt
    DATA_DIR_GPS = args.data_dir_gps
    RONIN_OUTPUTS_BASE_DIR = args.ronin_outputs_base_dir
    MODEL_SAVE_DIR = args.model_save_dir
    FIGURE_SAVE_DIR = args.figure_save_dir
    FULL_MODEL_PATH = osp.join(MODEL_SAVE_DIR, args.model_name)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)

    # --- Data Collection for Training ---
    print("\n--- Collecting and Preparing Data for Temporal Fusion Model Training ---")
    all_model_inputs_for_training_raw = []
    all_gt_positions_for_training_raw = []

    data_paths = []
    for data_name in os.listdir(DATA_DIR_GT):
        # Assuming RoNIN output files have the same base name as GT directories
        ronin_path = osp.join(RONIN_OUTPUTS_BASE_DIR, data_name + '_lstm_bi.npy')
        # Assuming GPS data is in a subfolder named after the data_name
        gps_path = osp.join(DATA_DIR_GPS, data_name, 'data_gps.csv')
        
        if osp.exists(ronin_path) and osp.exists(gps_path):
            data_paths.append((ronin_path, gps_path))

    if not data_paths:
        print(f"Error: No data found. Please check paths:\nGT: {DATA_DIR_GT}\nGPS: {DATA_DIR_GPS}\nRoNIN: {RONIN_OUTPUTS_BASE_DIR}")
        exit()

    for i, (ronin_output_path, gps_data_path) in enumerate(data_paths):
        processor_for_collection = DataProcessor(ronin_output_path, gps_data_path, IMU_SAMPLING_RATE)
        
        ronin_vel_pred, gt_pos, _ = processor_for_collection.load_and_extract_ronin_data()
        df_gps = processor_for_collection.load_gps_data()
        num_frames = ronin_vel_pred.shape[0] # RoNIN length defines master_timestamps
        master_timestamps = processor_for_collection.generate_master_timestamps(num_frames)

        model_input, gt_target = processor_for_collection.synchronize_data(
            df_gps, master_timestamps, ronin_vel_pred, gt_pos)

        all_model_inputs_for_training_raw.append(model_input)
        all_gt_positions_for_training_raw.append(gt_target)
        print(f"Collected data for training: {os.path.basename(os.path.dirname(gps_data_path))} ({i+1}/{len(data_paths)})")

    # # --- Normalization Step ---
    # # --- Normalization ---
    # # NORM_STATS_PATH will store ALL normalization stats (input and output)
    # NORM_STATS_PATH = './saved_models/normalization_stats.npz'

    # # 1. Create a single NormalizationManager instance
    # normalizer = NormalizationManager()

    # # 2. Fit this single instance using BOTH raw inputs and raw ground truth
    # normalizer.fit(all_model_inputs_for_training_raw, all_gt_positions_for_training_raw)

    # # 3. Save ALL calculated statistics to a single .npz file
    # normalizer.save_stats(NORM_STATS_PATH)

    # # 4. Apply normalization to the training inputs using transform_inputs()
    # all_model_inputs_for_training = [normalizer.transform_inputs(x) for x in all_model_inputs_for_training_raw]

    # # 5. Apply normalization to the ground truth targets using transform_outputs()
    # all_gt_positions_for_training = [normalizer.transform_outputs(x) for x in all_gt_positions_for_training_raw]


    # fusion_dataset = FusionDataset(all_model_inputs_for_training, all_gt_positions_for_training, SEQUENCE_LENGTH)
    # fusion_dataloader = DataLoader(fusion_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # # --- Training Loop ---
    # print(f"\n--- Training Temporal Fusion Model for {NUM_EPOCHS} Epochs ---")
    # temporal_fusion_model = TemporalFusionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_PROB).to(DEVICE)
    # optimizer = optim.Adam(temporal_fusion_model.parameters(), lr=LEARNING_RATE)
    # criterion = nn.MSELoss()
    # epoch_losses = [] 
    
    # temporal_fusion_model.train()

    # for epoch in range(NUM_EPOCHS):
    #     total_loss = 0
    #     num_batches = 0
    #     for batch_idx, (inputs, targets) in enumerate(fusion_dataloader):
    #         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    #         optimizer.zero_grad()
    #         fused_positions = temporal_fusion_model(inputs) # Model now outputs fused positions
    #         loss = criterion(fused_positions, targets) # Direct comparison to GT position
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         num_batches += 1

    #     avg_loss = total_loss / num_batches if num_batches > 0 else 0
    #     epoch_losses.append(avg_loss)
    #     print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.6f}")
    
    # print("Temporal Fusion Model Training Complete.")

    # # Save the ENTIRE trained model
    # torch.save(temporal_fusion_model, FULL_MODEL_PATH)
    # print(f"Full Temporal Fusion Model saved to: {FULL_MODEL_PATH}")

    # # --- Plotting the training loss ---
    # if args.plot_loss:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses, marker='o', linestyle='-', color='blue')
    #     plt.title('Temporal Fusion Model Training Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Average Loss')
    #     plt.grid(True)
    #     plt.xticks(range(1, NUM_EPOCHS + 1))
    #     loss_plot_path = osp.join(FIGURE_SAVE_DIR, 'loss_vs_epoch_plot.png') 
    #     plt.savefig(loss_plot_path)
    #     print(f"Training loss plot saved to: {loss_plot_path}")
    #     plt.close()

    # print("\n--- Training Script Finished ---")