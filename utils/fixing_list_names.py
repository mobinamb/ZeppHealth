import os

# Get directory names
data_dir_train = 'data/train/GT-data'
dir_names_train = os.listdir(data_dir_train)

# Write to the test list file
with open('lists/list_test_seen.txt', 'w') as f:
    for name in dir_names_train:
        f.write(name + '\n')


        
data_dir_test = 'data/test/GT-data'
dir_names_test = os.listdir(data_dir_test)

# Write to the test list file
with open('lists/list_test_unseen.txt', 'w') as f:
    for name in dir_names_test:
        f.write(name + '\n')