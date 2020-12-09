import numpy as np
import pickle as pkl
import json

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

def create_new_encoding(data, data_name):
    formatted_data = {}

    for j, protein_seq in enumerate(data):
        protein_length = 0
        secondary_structure_onehot = []
        primary_structure = ""
        secondary_structure = ""
        new_encoding = []

        for i in range(700):

            acid_start = i * 57
            acid_end = (i + 1) * 57

            acid_properties = protein_seq[acid_start:acid_end]

            one_hot_encoding_acid = acid_properties[:22]
            one_hot_encoding_label = acid_properties[22:31]

            PSSM = acid_properties[35:56]

            acid_id = np.argmax(one_hot_encoding_acid)
            label_id = np.argmax(one_hot_encoding_label)

            # Add 1e-4 for numerical stability
            one_dim_conservation = [np.log(21) + np.sum(PSSM * np.log(PSSM + 1e-4))]

            new_encoding += list(one_hot_encoding_acid[:22])
            new_encoding += list(ppDict[id_to_acid[acid_id]])
            new_encoding += list(PSSM)
            new_encoding += list(one_dim_conservation)

            secondary_structure_onehot += list(one_hot_encoding_label[:9])

            if acid_id == noseq_id_acid and label_id == noseq_id_label:
                pass
            else:
                primary_structure += id_to_acid[acid_id]
                secondary_structure += id_to_label[label_id]
                protein_length += 1

        formatted_data[j] = {}
        formatted_data[j]["protein_encoding"] = new_encoding
        formatted_data[j]["protein_length"] = protein_length
        formatted_data[j]["secondary_structure_onehot"] = secondary_structure_onehot
        formatted_data[j]["secondary_structure"] = secondary_structure
        formatted_data[j]["primary_structure"] = primary_structure

        if j % 100 == 0:
            print(j)
        print(primary_structure)

    return formatted_data


# %%

TR5534_data = np.load("../data/cullpdb+profile_5926.npy")
CB513_data = np.load("../data/cb513+profile_split1.npy")

# %%

cb513_formatted = create_new_encoding(CB513_data, "CB513")

# %%

tr5534_formatted = create_new_encoding(TR5534_data, "TR5534")

# %%

with open('TR5534.json', 'w') as outfile:
    json.dump(tr5534_formatted, outfile)

# %%

with open('CB513.json', 'w') as outfile:
    json.dump(cb513_formatted, outfile)

# %%


