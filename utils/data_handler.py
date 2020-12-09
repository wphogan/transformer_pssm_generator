import torch.utils.data as data

from utils.utils import *


class ProteinDataset(data.Dataset):
    def __init__(self, protein_data, ids, size, hp, device):
        self.hp = hp
        self.device = device
        data_len = len(ids)

        all_lengths = []
        protein_data_len = 22 + 7  # 22 one hot (21 acid types + 1 padding) + 7 ppDict
        all_poteins = []
        all_pssm = np.zeros([data_len, hp.max_seq_len, hp.num_features - protein_data_len])

        for i, id in enumerate(ids):
            id = str(id)
            if i % 250 == 0:
                print("Loading {0}/{1} proteins".format(i, len(ids)))

            d = protein_data[id]
            protein_length = d["protein_length"]
            protein_length = np.clip(protein_length, -0, hp.max_seq_len)
            all_lengths.append(protein_length)

            all_encodings = np.reshape(np.asarray(d["protein_encoding"]), (700, self.hp.num_features))
            acid_indices = np.argmax(all_encodings[:hp.max_seq_len, :22], axis=1)

            all_encodings = np.expand_dims(all_encodings, axis=0)
            all_poteins.append(acid_indices)
            all_pssm[i, :, :] = all_encodings[:, :hp.max_seq_len, 29:]

        self.all_poteins = torch.from_numpy(np.asarray(all_poteins).astype(np.int64))
        self.all_pssm = torch.from_numpy(all_pssm.astype(np.float32))
        self.all_lengths = torch.from_numpy(np.array(all_lengths).astype(np.int64))

        # Mean / std of PSSMs and lengths:
        pssm_mean_std = torch.std_mean(self.all_pssm, dim=(0, 1))
        lens_mean_std = torch.std_mean(self.all_lengths.float())
        print('PSSM mean/std:', pssm_mean_std)
        print('Seq len min: ', torch.min(self.all_lengths.float()))
        print('Seq len max: ', torch.max(self.all_lengths.float()))

        # Normalize PSSMs
        self.all_pssm = normalize(self.all_pssm, self.device)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        pssm = self.all_pssm[index]
        proteins = self.all_poteins[index]
        length = self.all_lengths[index]

        return proteins, pssm, length

    def __len__(self):
        return len(self.all_poteins)


def get_loader(protein_data, ids, hp, device, shuffle):
    """Returns torch.utils.data.DataLoader"""
    protein = ProteinDataset(protein_data, ids, hp=hp, device=device, size=(700 * hp.num_features))
    data_loader = torch.utils.data.DataLoader(dataset=protein,
                                              batch_size=hp.batch_size,
                                              shuffle=shuffle,
                                              num_workers=hp.num_workers)
    return data_loader
