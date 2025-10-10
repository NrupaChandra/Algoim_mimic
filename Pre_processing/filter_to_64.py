import pandas as pd
from pathlib import Path

output_path = Path(r"C:\Git\Algoim_mimic\Pre_processing\1MTestBernstein_p1_output_8.txt")
input_path  = Path(r"C:\Git\Algoim_mimic\Pre_processing\1MTestBernstein_p1_data.txt")

out_dir = Path(r"C:\Git\Algoim_mimic\Pre_processing")

df_out = pd.read_csv(output_path, sep=';', dtype=str)
df_in  = pd.read_csv(input_path,  sep=';', dtype=str)

def length_of_csv_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.strip()
    return s.eq("").map({True: 0, False: 1}) + s.str.count(',')

len_x = length_of_csv_series(df_out['nodes_x'])
len_y = length_of_csv_series(df_out['nodes_y'])
len_w = length_of_csv_series(df_out['weights'])

mask_64 = (len_x == 64) & (len_y == 64) & (len_w == 64)
df_out_64 = df_out[mask_64].copy()

keep_ids = set(df_out_64['id'].tolist())
df_in_64 = df_in[df_in['id'].isin(keep_ids)].copy()

filtered_output_path = out_dir / (output_path.stem + "_filtered64" + output_path.suffix)
filtered_input_path  = out_dir / (input_path.stem  + "_filtered64" + input_path.suffix)
ids_list_path        = out_dir / "filtered64_ids.txt"

df_out_64.to_csv(filtered_output_path, sep=';', index=False)
df_in_64.to_csv(filtered_input_path,  sep=';', index=False)
with open(ids_list_path, "w", encoding="utf-8") as f:
    for _id in sorted(keep_ids):
        f.write(f"{_id}\n")

print(f"Output rows total: {len(df_out)} | kept (64): {len(df_out_64)}")
print(f"Input rows total:  {len(df_in)}  | kept by id: {len(df_in_64)}")
print(f"Saved:\n  {filtered_output_path}\n  {filtered_input_path}\n  {ids_list_path}")