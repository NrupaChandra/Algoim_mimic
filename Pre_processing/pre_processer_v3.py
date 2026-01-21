#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch


def _parse_csv_floats(value: str) -> np.ndarray:
    return np.array(
        [float(v) for v in value.split(',') if v.strip() != ''],
        dtype=np.float32
    )


def preprocess_scales_and_centers(
    input_file: str,
    geom_file: str,
    save_dir: str = r'C:\Git\Algoim_mimic\Pre_processing\1kpreprocessed_chunks_scale_center',
    chunksize: int = 50000,
):
    """
    Saves per sample:
      (exp_x, exp_y, coeff,
       xscales, xcenters,
       yscales, ycenters,
       sample_id)
    """

    os.makedirs(save_dir, exist_ok=True)

    input_reader = pd.read_csv(input_file, sep=';', chunksize=chunksize)
    geom_reader  = pd.read_csv(geom_file,  sep=';', chunksize=chunksize)

    chunk_idx = 0
    chunk_files = []

    for in_chunk, geom_chunk in zip(input_reader, geom_reader):

        # Ensure alignment via ID
        if not in_chunk['id'].equals(geom_chunk['id']):
            in_chunk   = in_chunk.set_index('id')
            geom_chunk = geom_chunk.set_index('id')
            merged = in_chunk.join(geom_chunk, how='inner').reset_index()
        else:
            merged = pd.concat(
                [
                    in_chunk.reset_index(drop=True),
                    geom_chunk[['xscales', 'yscales','xcenters', 'ycenters']]
                        .reset_index(drop=True)
                ],
                axis=1
            )

        data_list = []

        for _, row in merged.iterrows():
            exp_x = torch.tensor(_parse_csv_floats(row['exp_x']), dtype=torch.float32)
            exp_y = torch.tensor(_parse_csv_floats(row['exp_y']), dtype=torch.float32)
            coeff = torch.tensor(_parse_csv_floats(row['coeff']), dtype=torch.float32)

            xscales  = torch.tensor(_parse_csv_floats(row['xscales']),  dtype=torch.float32)
            xcenters = torch.tensor(_parse_csv_floats(row['xcenters']), dtype=torch.float32)
            yscales  = torch.tensor(_parse_csv_floats(row['yscales']),  dtype=torch.float32)
            ycenters = torch.tensor(_parse_csv_floats(row['ycenters']), dtype=torch.float32)

            sample_id = row['id']

            data_list.append(
                (
                    exp_x, exp_y, coeff,
                    xscales, 
                    yscales,
                    xcenters, 
                    ycenters,
                    sample_id
                )
            )

        chunk_path = os.path.join(save_dir, f'preprocessed_chunk{chunk_idx}.pt')
        torch.save(data_list, chunk_path)
        chunk_files.append(chunk_path)

        print(f"Saved chunk {chunk_idx} with {len(data_list)} samples → {chunk_path}")
        chunk_idx += 1

    # Write index file
    index_path = os.path.join(save_dir, 'index.txt')
    with open(index_path, 'w') as f:
        for p in chunk_files:
            f.write(p + '\n')

    print(f"Index file saved to {index_path}")


if __name__ == "__main__":
    input_file = r'C:\Git\Algoim_mimic\Pre_processing\text_data\1kTestBernstein_p1_data_filtered64.txt'
    geom_file  = r'C:\Git\Algoim_mimic\Pre_processing\text_data\1kTestBernstein_p1_ScaleCenter.txt'

    preprocess_scales_and_centers(input_file, geom_file)
