import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from sklearn.model_selection import train_test_split

'''
CLASSES
'''
class Log:
    def __init__(self, debug=False, model_name='GAN'):
        time_stamp = datetime.now().strftime('%m%d_%H%M%S')
        self.model_name = model_name
        self.ts = time_stamp
        self.debug = debug
        self.main_dir = self.init_dirs()
        self.logger = self.init_logger()
        self.__call__("Experiment #: {}".format(self.ts))

    def __call__(self, msg):
        self.logger.info(msg)
        print(msg)

    def init_logger(self):
        log_filename = '{}/experiment.log'.format(self.main_dir, self.ts)
        logger = logging.getLogger(self.model_name)
        hdlr = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
        return logger

    def init_dirs(self):
        main_dir = 'experiments/{}'.format(self.ts)
        if self.debug:
            main_dir = 'experiments/debug'
        dirs = [
            main_dir
        ]
        make_directory(dirs)
        return main_dir


'''
FUNCTIONS
'''
def get_acid_arr():
    return np.array(['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V',
                     'Y', 'X', '_'])  # _ = padding


def get_acid_dict():
    acid_to_id = {
        'A': 0, 'C': 1, 'E': 2, 'D': 3, 'G': 4, 'F': 5, 'I': 6, 'H': 7, 'K': 8, 'M': 9, 'L': 10, 'N': 11,
        'Q': 12, 'P': 13, 'S': 14, 'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19, 'X': 20, '_': 21
    }
    return acid_to_id


def get_mean_and_std():
    '''Calculated from TR5534'''
    mean = torch.tensor([0.2443, 0.1649, 0.2440, 0.2506, 0.2120, 0.2064, 0.2191, 0.2392, 0.2353,
                         0.2348, 0.2216, 0.2266, 0.2010, 0.2363, 0.2354, 0.2277, 0.2253, 0.2388,
                         0.1796, 0.1549, 0.2081, 1.9335])
    std = torch.tensor([0.1123, 0.0505, 0.0957, 0.1038, 0.0743, 0.0722, 0.0874, 0.0922, 0.0969,
                        0.0903, 0.0907, 0.0929, 0.0649, 0.1020, 0.0947, 0.1028, 0.1006, 0.0969,
                        0.0506, 0.0919, 0.0740, 1.9039])
    return mean, std


def normalize(seqs, device='cpu'):
    mean, std = get_mean_and_std()
    bs = seqs.shape[0]
    for i in range(bs):
        s = (seqs[i, :, :].to(device) - mean.to(device)) / std.to(device)
        seqs[i, :, :] = s.unsqueeze(0)
    return seqs


def unnormalize(seqs, device):
    mean, std = get_mean_and_std()
    bs = seqs.shape[0]
    for i in range(bs):
        s = (seqs[i, :, :].to(device) * std.to(device)) + mean.to(device)
        seqs[i, :, :] = s.unsqueeze(0)
    return seqs


def splits(tr5534_data, debug, SEED):
    all_ids = np.arange(0, len(tr5534_data))
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=SEED)
    if debug:
        train_ids = train_ids[:3]
        val_ids = val_ids[:1]
    return train_ids, val_ids


def set_gen_ids(data, debug):
    ids = np.arange(0, len(data))  # load all ids
    if debug:
        return ids[0:5]  # short run
    else:
        return ids


def save_pssm(pssm, hp, log, filename=False):
    if not filename:
        filename = '{}/{}'.format(log.main_dir, hp.data_file_new_pssm)
    with open(filename, 'wb') as f:
        np.save(f, pssm)
    log('Saved new PSSM: {}'.format(filename))


def make_directory(directories):
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except Exception:
            print("Error -- could not create directory.")


def graph_losses(losses, log):
    outfile = '{}/loss_plot.png'.format(log.main_dir)
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Training Loss')
    plt.plot(losses.T[1], label='Validation Loss')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig(outfile)
    log('Saved plot: {}'.format(log.ts))
    plt.close()


def debug_warning(debug, log):
    pluses = '*' * 10
    if debug:
        log('\n' + pluses + ' WARNING, DEBUG MODE ' + pluses)
    else:
        log('\n' + pluses + ' FULL RUN ' + pluses)


def set_device(log):
    if torch.cuda.is_available():
        log("Nice! Using GPU.")
        return 'cuda'
    else:
        log("Watch out! Using CPU.")
        return 'cpu'


def memory_usage():
    torch.cuda.empty_cache()
    cuda_allocated = torch.cuda.memory_allocated() / 1024 ** 2
    cuda_cahced = torch.cuda.memory_cached() / 1024 ** 2
    cpu_avail = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    print('Pod stats >> CUDA Allctd: {} | CUDA Cached: {} | CPU Avail: {} '.format(cuda_allocated, cuda_cahced,
                                                                                   cpu_avail))


def save_model(model, optim, log):
    if log.debug: return  # Don't save model in debug mode
    model_dict = {
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
    }
    location = '{}/model_dict.bin'.format(log.main_dir)
    torch.save(model_dict, location)
    log('Saved model.')


def load_model(path, model, device, log):
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    log('Model loaded: ' + path)
    return model


def trim_padding(output, trg, lens, device):
    new_out, new_trg = [], []
    output_dim = output.shape[-1]
    for i in range(len(lens)):
        o_trim = output[i, 1:lens[i], :]
        new_out.append(o_trim.view(-1, output_dim))
        t_trim = trg[i, 1:lens[i], :]
        new_trg.append(t_trim.view(-1, output_dim))
    new_out = torch.cat(new_out).contiguous().to(device)
    new_trg = torch.cat(new_trg).contiguous().to(device)
    return new_out, new_trg
