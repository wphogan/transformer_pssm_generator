# %%

import numpy as np
import json
# %%

data = []
with open("../data/TR6614-fasta_labels2.txt", "r") as f:
    for line in f:
        data.append(line.rstrip("\n"))

# %%

data = np.array(data)

# %%

inds = np.where(data == '')

# %%

data = np.delete(data, inds)

# %%

start = 0
end = 19840

# %%

pmap = {"X": [],
        "Y": []}

# %%

for i in np.arange(start, end, step=3):
    title = data[i]
    proteins = data[i + 1]
    labels = data[i + 2]

    pmap["X"].append(proteins)
    pmap["Y"].append(labels)

# %%

pmap["X"]

# %%

id_to_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X',
              'pad']
id_to_label = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'pad']

ppDict = {"A": [1.28, 0.05, 1.0, 0.31, 6.11, 0.42, 0.23], "G": [0.00, 0.00, 0.0, 0.00, 6.07, 0.13, 0.15],
          "V": [3.67, 0.14, 3.0, 1.22, 6.02, 0.27, 0.49], "L": [2.59, 0.19, 4.0, 1.70, 6.04, 0.39, 0.31],
          "I": [4.19, 0.19, 4.0, 1.80, 6.04, 0.30, 0.45], "F": [2.94, 0.29, 5.89, 1.79, 5.67, 0.3, 0.38],
          "Y": [2.94, 0.3, 6.47, 0.96, 5.66, 0.25, 0.41], "W": [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
          "T": [3.03, 0.11, 2.60, 0.26, 5.6, 0.21, 0.36], "S": [1.31, 0.06, 1.6, -0.04, 5.7, 0.20, 0.28],
          "R": [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25], "K": [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
          "H": [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.3], "D": [1.6, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
          "E": [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21], "N": [1.6, 0.13, 2.95, -0.6, 6.52, 0.21, 0.22],
          "Q": [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25], "M": [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
          "P": [2.67, 0.0, 2.72, 0.72, 6.8, 0.13, 0.34], "C": [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
          "X": [0, 0, 0, 0, 0, 0, 0], "pad": [0, 0, 0, 0, 0, 0, 0]}

protein_to_id = {id_to_acid[i]: i for i in range(len(id_to_acid))}
label_to_id = {id_to_label[i]: i for i in range(len(id_to_label))}
noseq_id_acid = 21
noseq_id_label = 8


# %%

def create_new_encoding(pmap):
    formatted_data = {}

    for i in range(len(pmap["X"])):
        secondary_structure_onehot = []
        primary_structure = ""
        secondary_structure = ""
        new_encoding = []

        primary_structure = pmap["X"][i]
        secondary_structure = pmap["Y"][i]

        protein_length = len(primary_structure)
        padding_length = 700 - protein_length

        for x, y in zip(primary_structure, secondary_structure):
            primary_onehot = [0] * 22
            secondary_onehot = [0] * 9

            primary_onehot[protein_to_id[x]] = 1
            secondary_onehot[label_to_id[y]] = 1

            acid_properties = ppDict[x]
            pssm_and_conservation = [0] * 22

            new_encoding += primary_onehot
            new_encoding += list(acid_properties)
            new_encoding += pssm_and_conservation
            secondary_structure_onehot += secondary_onehot

        for _ in range(padding_length):
            primary_onehot = [0] * 22
            primary_onehot[-1] = 1

            secondary_onehot = [0] * 9
            secondary_onehot[-1] = 1

            acid_properties = list(ppDict["pad"])
            pssm_and_conservation = [0] * 22

            new_encoding += primary_onehot
            new_encoding += list(acid_properties)
            new_encoding += pssm_and_conservation
            secondary_structure_onehot += secondary_onehot

        if i % 100 == 0:
            print(i)

        i = str(i)
        formatted_data[i] = {}
        formatted_data[i]["protein_encoding"] = new_encoding
        formatted_data[i]["protein_length"] = protein_length
        formatted_data[i]["secondary_structure_onehot"] = secondary_structure_onehot
        formatted_data[i]["secondary_structure"] = secondary_structure
        formatted_data[i]["primary_structure"] = primary_structure

    return formatted_data


# %%

formatted_data = create_new_encoding(pmap)

# %%

with open('../data/TR6614_no_pssm_onedim.json', 'w') as outfile:
    json.dump(formatted_data, outfile)

# %%



# %%

