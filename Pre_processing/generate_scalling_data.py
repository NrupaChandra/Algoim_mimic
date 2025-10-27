import numpy as np


file_path = r"C:\Git\Algoim_mimic\Pre_processing\1kTestBernstein_p1_output_8_filtered64.txt"
output_file = r"C:\Git\Algoim_mimic\Pre_processing\1kTestBernstein_p1_Weight_scalled.txt"

#weightRefs = np.array([0.1012285362903763 ,0.2223810344533745 ,0.3137066458778873 ,0.3626837833783620 ,0.3626837833783620 ,0.3137066458778873 ,0.2223810344533745, 0.1012285362903763])
deltaRef = 0.9602898564975363*2 #always same since it comes from fixed quadrature 
#w = np.outer(weightRefs,weightRefs)

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
                nodes_x_all.append(x)
                nodes_y_all.append(y)
            buffer = "" 

    nodes_x_all = [x.reshape(8, 8) for x in nodes_x_all]
    nodes_y_all = [y.reshape(8, 8) for y in nodes_y_all]


    return ids, nodes_x_all, nodes_y_all


ids, nodes_x_all, nodes_y_all = read_predicted_data(file_path)

xdeltas = np.stack([nx[:, 7] - nx[:, 0] for nx in nodes_x_all])
ydeltas = np.stack([ny[7, :] - ny[0, :] for ny in nodes_y_all])

xscales = xdeltas/deltaRef
yscales = ydeltas/deltaRef

print (xscales.shape)
print("x deltas for the first node set are", xdeltas[0])
print("x scales for the first node set are", xscales[0])


with open(output_file, "w", encoding="utf-8") as f:
    f.write("number;id;xscales;yscales\n")
    for idx, (id_str, xs, ys) in enumerate(zip(ids, xscales, yscales), start=1):
        x_str = ",".join(map(str, xs))
        y_str = ",".join(map(str, ys))
        f.write(f"{idx};{id_str};{x_str};{y_str}\n")

print(f"Saved scaling factors to {output_file}")

