#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch

def _parse_csv_floats(value: str) -> np.ndarray:
    return np.array([float(v) for v in value.split(',') if v.strip() != ''], dtype=np.float32)

def preprocess_scales(
    input_file: str,
    scales_file: str,
    save_dir: str = r'C:\Git\Algoim_mimic\Pre_processing\1kpreprocessed_chunks_weight_scaled',
    chunksize: int = 50000,
    compute_2d_scales: bool = True
):
    """
    Reads input (exp_x, exp_y, coeff) and output (xscales, yscales) files in chunks,
    converts each row to torch tensors, and saves per-chunk .pt files.

    Saved sample tuple per row:
      (exp_x, exp_y, coeff, xscales, yscales, scales2d_or_None, sample_id)
    """
    os.makedirs(save_dir, exist_ok=True)

    input_reader  = pd.read_csv(input_file,  sep=';', chunksize=chunksize)
    scales_reader = pd.read_csv(scales_file, sep=';', chunksize=chunksize)

    chunk_idx = 0
    chunk_files = []

    for in_chunk, sc_chunk in zip(input_reader, scales_reader):
        # Optional sanity check that IDs align (cheap & defensive)
        if not in_chunk['id'].equals(sc_chunk['id']):
            # If files are guaranteed aligned and same order, you can drop this.
            # Otherwise, join on 'id' to be safe:
            in_chunk  = in_chunk.set_index('id')
            sc_chunk  = sc_chunk.set_index('id')
            merged = in_chunk.join(sc_chunk, how='inner', lsuffix='_in', rsuffix='_sc').reset_index()
        else:
            merged = pd.concat([in_chunk.reset_index(drop=True), 
                                sc_chunk[['xscales','yscales']].reset_index(drop=True)], axis=1)

        data_list = []
        for _, row in merged.iterrows():
            exp_x   = torch.tensor(_parse_csv_floats(row['exp_x']), dtype=torch.float32)
            exp_y   = torch.tensor(_parse_csv_floats(row['exp_y']), dtype=torch.float32)
            coeff   = torch.tensor(_parse_csv_floats(row['coeff']), dtype=torch.float32)

            xscales = torch.tensor(_parse_csv_floats(row['xscales']), dtype=torch.float32)  # shape (8,)
            yscales = torch.tensor(_parse_csv_floats(row['yscales']), dtype=torch.float32)  # shape (8,)

            if compute_2d_scales:
                # Outer product → (8,8) → flatten to (64,) for compact storage
                scales2d = torch.tensor(
                    np.outer(xscales.numpy(), yscales.numpy()).astype(np.float32).ravel(),
                    dtype=torch.float32
                )
            else:
                scales2d = None

            sample_id = row['id']

            data_list.append((exp_x, exp_y, coeff, xscales, yscales, scales2d, sample_id))

        chunk_path = os.path.join(save_dir, f'preprocessed_chunk{chunk_idx}.pt')
        torch.save(data_list, chunk_path)
        chunk_files.append(chunk_path)
        print(f"Saved chunk {chunk_idx} with {len(data_list)} samples to {chunk_path}")
        chunk_idx += 1

    # Write an index of chunk files
    index_path = os.path.join(save_dir, 'index.txt')
    with open(index_path, 'w') as f:
        for p in chunk_files:
            f.write(p + '\n')
    print(f"Index file saved to {index_path}")

if __name__ == "__main__":
    input_file  = r'C:\Git\Algoim_mimic\Pre_processing\text_data\1kTestBernstein_p1_data_filtered64.txt'
    scales_file = r'C:\Git\Algoim_mimic\Pre_processing\text_data\1kTestBernstein_p1_Weight_scalled.txt'
    preprocess_scales(input_file, scales_file, compute_2d_scales=True)
