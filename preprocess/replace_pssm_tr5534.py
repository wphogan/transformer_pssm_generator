import json

import numpy as np

print('starting. loading data...')
train_data = json.load(open('../data/TR5534.json', "r"))
# train_data_new = json.load(open('../data/TR5534_aug_transformer_v3_whole.json', "r"))
# python train.py --experiment base_tr5534_solo_aug > base_tr5534_solo_aug_train.out
print()
pssm_data = np.load("../experiments/1204_090945/tr5534_whole_gen_pssm.npy") # TODO MUST BE GENERATED, UNSHUFFLED TR5534 PSSM DATA
outfile_dest = '../data/TR5534_aug_transformer_v3_whole.json'
updated_data = {}
print('data loaded...')
i = -1
for line, pssm in zip(train_data, pssm_data):
    orig_encoding = train_data[line]['protein_encoding']
    orig_flattened = orig_encoding
    orig_encoding = np.reshape(np.asarray(orig_encoding), (700, 51))
    orig_encoding[:,29:] = pssm[:,:]

    # Messy iterator
    i += 1
    if i % 100 == 0:
        print('Processing line ', i)

    # Save newly replaced PSSMs
    train_data[line]["protein_encoding"] = list(orig_encoding.flatten().astype(float))


with open(outfile_dest, 'w') as outfile:
    json.dump(train_data, outfile)

print('complete. save fle:', outfile_dest)