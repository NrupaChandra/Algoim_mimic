import numpy as np


file_path = r"C:\Git\Algoim_mimic\FNN\Results\predicted_data_fnn.txt"

weightRefs = np.array([0.1012285362903763 ,0.2223810344533745 ,0.3137066458778873 ,0.3626837833783620 ,0.3626837833783620 ,0.3137066458778873 ,0.2223810344533745, 0.1012285362903763])
deltaRef = 0.9602898564975363*2 #always same since it comes from fixed quadrature 

#tensor product of weights
w = np.outer(weightRefs,weightRefs)
#print(np.sum(w)) 
#print(w[0,0])

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
                x = np.fromstring(x_str.strip()[1:-1], sep=" ")
                y = np.fromstring(y_str.strip()[1:-1], sep=" ")
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

'''print("xvals for the first node set are ", nodes_x_all[0])
print("x deltas for the first node set are", xdeltas[0])
print("x scales for the first node set are", xscales[0])
 '''

scaled_weights_all = []

for k in range(len(ids)):
    scaled = (xscales[k][:, None] * yscales[k][None, :]) * w
    scaled_weights_all.append(scaled)


'''
print(scaled_weights_all[0].shape)   
print("Scaled weights for first node set:\n", scaled_weights_all[0])

s = np.sum(scaled_weights_all[0])
print("sum of the first set is :", s)'''


output_file = r"C:\Git\Algoim_mimic\Weight_scalling\Weight_scalling.txt"

with open(output_file, "w") as f:
    f.write("number;id;nodes_x;nodes_y;weights\n\n")
    
    for idx, (id_str, x, y, w) in enumerate(zip(ids, nodes_x_all, nodes_y_all, scaled_weights_all), start=1):
        # flatten arrays to 1D strings
        x_flat = " ".join(map(str, x.flatten()))
        y_flat = " ".join(map(str, y.flatten()))
        w_flat = " ".join(map(str, w.flatten()))
        
        f.write(f"number: {idx}\n")
        f.write(f"id: {id_str}\n")
        f.write(f"nodes_x: [{x_flat}]\n")
        f.write(f"nodes_y: [{y_flat}]\n")
        f.write(f"weights: [{w_flat}]\n")
        f.write("\n")  # blank line between records

print(f"Saved scaled data to {output_file}")
