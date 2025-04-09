import random

# File paths

input_file = "filenames/sceneflow_driving_all_asus_pc.txt"
train_file = "filenames/sceneflow_driving_train_asus_pc.txt"
test_file = "filenames/sceneflow_driving_test_asus_pc.txt"
valid_file = "filenames/sceneflow_driving_valid_asus_pc.txt"

# Read all lines from the input file
with open(input_file, "r") as file:
    lines = file.readlines()

# Shuffle and select 20% of the lines
random.shuffle(lines)
num_selected = len(lines) // 5  # 20% of the lines
selected_lines = lines[:num_selected]

# Split the selected lines into two halves
half = len(selected_lines) // 2
test_lines = selected_lines[:half]
valid_lines = selected_lines[half:]

# Write the selected lines to the test and validation files
with open(test_file, "w") as file:
    file.writelines(test_lines)

with open(valid_file, "w") as file:
    file.writelines(valid_lines)

# Remove the selected lines from the original file
remaining_lines = lines[num_selected:]
with open(train_file, "w") as file:
    file.writelines(remaining_lines)

print(f"Moved {len(test_lines)} lines to {test_file}")
print(f"Moved {len(valid_lines)} lines to {valid_file}")
print(f"Remaining lines in {train_file}: {len(remaining_lines)}")