import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pred_gt = np.load('outputs/resnet_lstm/a000_7_lstm_bi.npy')
gps = pd.read_csv('data/synthetic-GPS/seen_subjects_test_set/a000_7/data_gps.csv', header=None).values

print(gps[1])
# Plotting the predicted and ground truth GPS data
plt.figure(figsize=(10, 6))
x_m = [np.float32(gps[i][0]) for i in range(1,len(gps))]
y_m = [np.float32(gps[i][1]) for i in range(1,len(gps))]
time_stamp = [np.float32(gps[i][2]) for i in range(1,len(gps))]
HDOP = [np.float32(gps[i][3]) for i in range(1,len(gps))]

print([pred_gt[0]])
print(gps[0])

# plt.plot(x_m, y_m, 'r.', label='GPS')
sc = plt.scatter(x_m, y_m, c=HDOP, cmap='viridis', s=20, edgecolor='k', label='GPS')
plt.plot(pred_gt[:, 0], pred_gt[:, 1], 'b.', label='Predicted GPS')
plt.plot(pred_gt[:, 2], pred_gt[:, 3], 'g.', label='Ground Truth GPS')
cbar = plt.colorbar(sc)
cbar.set_label('HDOP')
plt.xlabel('x_m')
plt.ylabel('y_m')
plt.title('IMU Predicted IMU vs synthetic GPS vs GrandTruth Data')
plt.legend()
plt.show()