import numpy as np


file_path = r"C:\Git\Algoim_mimic\Pre_processing\text_data\100kTestBernstein_p1_output_8_filtered64_sorted.txt"
output_file = r"C:\Git\Algoim_mimic\Pre_processing\text_data\100kTestBernstein_p1_ScaleCenter.txt"
deltaRef = 0.9602898564975363 * 2.0


# ------------------------------------------------------------
# Read nodes from file
# ------------------------------------------------------------
def read_predicted_data(path):
    ids = []
    nodes_x_all = []
    nodes_y_all = []

    with open(path, "r") as f:
        text = f.read().strip()

    if text.lower().startswith("number;id;nodes_x"):
        text = text.split("\n", 1)[1]

    records = text.splitlines()

    buffer = ""
    for line in records:
        buffer += line.strip() + " "
        if buffer.count(";") >= 4:
            parts = buffer.split(";", 4)
            if len(parts) == 5:
                _, id_str, x_str, y_str, _ = parts
                x = np.fromstring(x_str.strip(), sep=",")
                y = np.fromstring(y_str.strip(), sep=",")
                ids.append(id_str.strip())
                nodes_x_all.append(x.reshape(8, 8))
                nodes_y_all.append(y.reshape(8, 8))
            buffer = ""

    return ids, nodes_x_all, nodes_y_all


ids, nodes_x_all, nodes_y_all = read_predicted_data(file_path)
N = len(ids)


# ------------------------------------------------------------
# Compute scales and centers
# ------------------------------------------------------------
# x: row-wise (varying in x-direction)
xdeltas  = np.stack([nx[:, 7] - nx[:, 0] for nx in nodes_x_all])     # (N, 8)
xcenters = np.stack([(nx[:, 7] + nx[:, 0]) * 0.5 for nx in nodes_x_all])

# y: column-wise (varying in y-direction)
ydeltas  = np.stack([ny[7, :] - ny[0, :] for ny in nodes_y_all])     # (N, 8)
ycenters = np.stack([(ny[7, :] + ny[0, :]) * 0.5 for ny in nodes_y_all])

xscales = xdeltas / deltaRef
yscales = ydeltas / deltaRef


# ------------------------------------------------------------
# Sanity check (first sample)
# ------------------------------------------------------------
print("x-scales shape :", xscales.shape)
print("y-scales shape :", yscales.shape)
print("x-centers shape:", xcenters.shape)
print("y-centers shape:", ycenters.shape)

print("\nFirst sample:")
print("x-scales :", xscales[0])
print("x-centers:", xcenters[0])
print("y-scales :", yscales[0])
print("y-centers:", ycenters[0])


# ------------------------------------------------------------
# Write output
# ------------------------------------------------------------
with open(output_file, "w", encoding="utf-8") as f:
    f.write("number;id;xscales;yscales;xcenters;ycenters\n")
    for k, (id_str, xs, ys, xc, yc) in enumerate(
        zip(ids, xscales, yscales, xcenters, ycenters), start=1
    ):
        xs_str = ",".join(map(str, xs))
        ys_str = ",".join(map(str, ys))
        xc_str = ",".join(map(str, xc))
        yc_str = ",".join(map(str, yc))
        f.write(f"{k};{id_str};{xs_str};{ys_str};{xc_str};{yc_str}\n")

print(f"\nSaved scale + center data to:\n{output_file}")
