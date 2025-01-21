import os
import json
import csv

results_dir = '.'
output_csv = 'results_summary.csv'

# Function to extract checkpoint name and exit layer from the folder path
def extract_info_from_path(path):
    parts = path.split('__')
    checkpoint_name = '__'.join(parts[-2:])
    exit_layer = path.split('/')[1].split('_')[1]
    return checkpoint_name, exit_layer

# List to store the results
results = []

# Traverse the results directory
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            checkpoint_name, exit_layer = extract_info_from_path(root)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract relevant data
                result = {'checkpoint_name': checkpoint_name, 'exit_layer': exit_layer}
                for key, value in data['results'].items():
                    if 'mmlu_' in key:
                        continue
                    else:
                        result[key] = value['acc,none']
                
                results.append(result)

# Get all dataset keys
all_keys = set()
for result in results:
    all_keys.update(result.keys())
all_keys.discard('checkpoint_name')
all_keys.discard('exit_layer')
all_keys = sorted(all_keys)

# Write results to CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['checkpoint_name', 'exit_layer'] + all_keys
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f'Results written to {output_csv}')