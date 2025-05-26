import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pyproj import Transformer
import os.path as osp
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse


from proposed_network_train import TemporalFusionModel
from proposed_network_train import DataProcessor
from proposed_network_train import NormalizationManager 
from metric import compute_relative_trajectory_error


# --- Main Evaluation Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Temporal Fusion Model on test data.")
    parser.add_argument('--model_path', type=str, default='./saved_models/temporal_fusion_model.pth',
                        help='Path to the saved trained model (.pth file).')
    parser.add_argument('--norm_stats_path', type=str, default='./saved_models/normalization_stats.npz',
                        help='Path to the saved normalization statistics (.npz file).')
    parser.add_argument('--data_dir_gt_test', type=str, default='./data/test/GT-data',
                        help='Path to the directory containing GT data for testing.')
    parser.add_argument('--data_dir_gps_test', type=str, default='./data/test/synthetic-GPS/modified_unseen_test_set/',
                        help='Path to the directory containing synthetic GPS data for testing.')
    parser.add_argument('--ronin_outputs_base_dir', type=str, default='./outputs/resnet_lstm_test/',
                        help='Base directory for RoNIN model outputs used in testing.')
    parser.add_argument('--figure_save_dir', type=str, default='./figures',
                        help='Directory to save evaluation plots.')
    parser.add_argument('--imu_sampling_rate', type=int, default=200,
                        help='IMU/RoNIN sampling rate in Hz (must match training).')
    parser.add_argument('--plot_trajectory_idx', type=int, default=0,
                        help='Index of the trajectory in the test set to plot (e.g., 0 for the first one).')
    parser.add_argument('--plot_all_trajectories', action='store_true',
                        help='If set, plots all trajectories in the test set instead of just one.')

    args = parser.parse_args()

    # --- Configuration ---
    INPUT_DIM = 6
    HIDDEN_DIM = 8 
    OUTPUT_DIM = 2
    NUM_LAYERS = 2
    DROPOUT_PROB = 0.1

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    os.makedirs(args.figure_save_dir, exist_ok=True)

    # --- Load the Trained Model ---
    if not osp.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}. Please train the model first.")
        exit()

    print(f"\n--- Loading pre-trained Temporal Fusion Model from: {args.model_path} ---")
    temporal_fusion_model = TemporalFusionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_PROB).to(DEVICE)
    temporal_fusion_model = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    temporal_fusion_model.eval()
    print("Full model loaded successfully.")

    # --- Load Normalization Statistics ---
    normalizer = NormalizationManager()
    try:
        normalizer.load_stats(args.norm_stats_path)
    except FileNotFoundError as e:
        print(f"Error loading normalization stats: {e}")
        exit()

    # --- Collect Test Data Paths ---
    print("\n--- Collecting Test Data ---")
    test_data_paths = []
    for data_name in os.listdir(args.data_dir_gt_test):
        ronin_path = osp.join(args.ronin_outputs_base_dir, data_name + '_lstm_bi.npy')
        gps_path = osp.join(args.data_dir_gps_test, data_name, 'data_gps.csv')
        if osp.exists(ronin_path) and osp.exists(gps_path):
            test_data_paths.append((ronin_path, gps_path))

    if not test_data_paths:
        print(f"Error: No test data found in {args.data_dir_gt_test} or {args.data_dir_gps_test}.")
        exit()

    print(f"Found {len(test_data_paths)} test trajectories.")

    # --- Process and Plot Trajectories ---
    trajectories_to_plot = []
    if args.plot_all_trajectories:
        trajectories_to_plot = test_data_paths
    elif 0 <= args.plot_trajectory_idx < len(test_data_paths):
        trajectories_to_plot = [test_data_paths[args.plot_trajectory_idx]]
    else:
        print(f"Invalid --plot_trajectory_idx {args.plot_trajectory_idx}. Must be between 0 and {len(test_data_paths)-1}.")
        exit()

    for idx, (sample_ronin_path, sample_gps_path) in enumerate(trajectories_to_plot):
        trajectory_name = os.path.basename(os.path.dirname(sample_gps_path))
        print(f"\n--- Processing Trajectory: {trajectory_name} (Index: {idx}) ---")

        processor_for_inference = DataProcessor(sample_ronin_path, sample_gps_path, args.imu_sampling_rate)

        ronin_vel_pred_inference, gt_pos_inference, _ = processor_for_inference.load_and_extract_ronin_data()
        df_gps_inference = processor_for_inference.load_gps_data()
        num_frames_inference = ronin_vel_pred_inference.shape[0]
        master_timestamps_inference = processor_for_inference.generate_master_timestamps(num_frames_inference)

        # model_input_inference will be the RAW data features
        model_input_raw, gt_target_raw = processor_for_inference.synchronize_data(
            df_gps_inference, master_timestamps_inference, ronin_vel_pred_inference, gt_pos_inference)

        # Apply normalization to the RAW input data before feeding to the model
        model_input_normalized = normalizer.transform_inputs(model_input_raw)

        # Run inference
        with torch.no_grad():
            # Add batch dimension for inference (model expects batch_size, seq_len, features)
            inference_input_tensor = torch.tensor(model_input_normalized, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            fused_positions_tensor_normalized = temporal_fusion_model(inference_input_tensor)

        # Denormalize the model's output back to original scale
        fused_positions_np = normalizer.denormalize_outputs(fused_positions_tensor_normalized[0, :, :].cpu().numpy())

        # --- Metrics Calculation ---
        # Compare denormalized predictions to the original RAW ground truth
        rmse = np.sqrt(np.mean((fused_positions_np - gt_target_raw)**2))
        rte = compute_relative_trajectory_error(fused_positions_np, gt_target_raw, delta=fused_positions_np.shape[0] - 1)

        print(f"RMSE of Fused Positions vs. GT for {trajectory_name}: {rmse:.4f} meters")
        print(f"RTE of Fused Positions vs. GT for {trajectory_name}: {rte:.4f} meters")

        # --- Plotting ---
        plt.figure(figsize=(12, 8))
        # Plot against the original RAW ground truth
        plt.plot(gt_target_raw[:, 0], gt_target_raw[:, 1], label='Ground Truth', color='black', linestyle='--', linewidth=2)
        plt.plot(fused_positions_np[:, 0], fused_positions_np[:, 1], label='Fused Position (Model Output)', color='red', linestyle='dotted', linewidth=1.5)
        plt.plot(model_input_raw[:, 0], model_input_raw[:, 1], label='Raw Synchronized GPS', color='blue', alpha=0.6, linestyle=':')

        # Plot RoNIN-integrated path
        ronin_path_integrated = np.zeros_like(gt_target_raw)
        ronin_path_integrated[0] = gt_target_raw[0]
        dt = 1.0 / args.imu_sampling_rate
        for i in range(1, num_frames_inference):
            ronin_path_integrated[i, 0] = ronin_path_integrated[i-1, 0] + ronin_vel_pred_inference[i, 0] * dt
            ronin_path_integrated[i, 1] = ronin_path_integrated[i-1, 1] + ronin_vel_pred_inference[i, 1] * dt
        plt.plot(ronin_path_integrated[:, 0], ronin_path_integrated[:, 1], label='RoNIN Integrated Path', color='green', linestyle='-.', alpha=0.7)

        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title(f'Trajectory Comparison: {trajectory_name}\nRMSE: {rmse:.4f}m')
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')

        plot_filename = f'fused_path_comparison_{trajectory_name}.png'
        fusion_plot_path = osp.join(args.figure_save_dir, plot_filename)
        plt.savefig(fusion_plot_path)
        print(f"Comparison plot saved to: {fusion_plot_path}")
        plt.close()

    print("\n--- Evaluation Script Finished ---")



