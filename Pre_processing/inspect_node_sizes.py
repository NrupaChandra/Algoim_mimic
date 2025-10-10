import pandas as pd

file_path = r"C:\Git\Algoim_mimic\Pre_processing\10kTestBernstein_p1_output_8_filtered64.txt"

df = pd.read_csv(file_path, sep=';')

# Check the first few rows and compute lengths
lengths = []
for idx, row in df.iterrows():
    nodes_x_len = len(row['nodes_x'].split(','))
    nodes_y_len = len(row['nodes_y'].split(','))
    weights_len = len(row['weights'].split(','))
    lengths.append((idx, nodes_x_len, nodes_y_len, weights_len))

# Print the first 50 entries
for entry in lengths[:50]:
    print(f"Row {entry[0]}: nodes_x={entry[1]}, nodes_y={entry[2]}, weights={entry[3]}")
