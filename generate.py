'''
Generate.py
This script loads a trained PSSM generation model and
uses it to generate PSSMs for protein sequences.
'''
import argparse

from models.transformer import Transformer, Encoder, Decoder
from utils.data_handler import *
from utils.train_utils import *
from utils.utils import *


class GenerateHyperparameters():
    def __init__(self):
        # Data:
        self.data_dir_no_pssm = 'data/TR6614_no_pssm_onedim.json'
        self.data_dir_tr = 'data/TR5534.json'

        # Experiment settings:
        self.batch_size = 16
        self.num_workers = 4
        self.lr = 5e-4
        self.num_features = 51

        # Transformer settings:
        self.max_seq_len = 700
        self.input_dim = 22  # acid types + pad
        self.output_dim = 22  # PSSM dim
        self.hid_dim = 256
        self.enc_layers = 3
        self.dec_layers = 3
        self.enc_heads = 8
        self.dec_heads = 8
        self.enc_pf_dim = 512
        self.dec_pf_dim = 512
        self.enc_dropout = 0.1
        self.dec_dropout = 0.1


def main(args):
    # Init
    hp = GenerateHyperparameters()
    log = Log(args.debug)
    device = set_device(log)

    # Experiment settings
    target_data = 'tr6614_whole'  # tr6614_whole tr5534_whole <<<<<< TOGGLE THIS
    log('target data: {}'.format(target_data))
    model_loc = 'experiments/1204_090945/model_dict.bin'
    output_pssm_file = 'experiments/1204_090945/{}_gen_pssm.npy'.format(target_data)

    tr5534, tr6614 = hp.data_dir_tr, hp.data_dir_no_pssm
    if target_data == 'tr5534_whole':
        gen_dir = tr5534
    elif target_data == 'tr6614_whole':
        gen_dir = tr6614
    else:
        raise Exception('Target dir not recognized.')

    # Model init
    encoder = Encoder(hp.input_dim, hp.hid_dim, hp.enc_layers, hp.enc_heads, hp.enc_pf_dim, hp.enc_dropout, device,
                      max_length=hp.max_seq_len)
    decoder = Decoder(hp.input_dim, hp.hid_dim, hp.dec_layers, hp.dec_heads, hp.dec_pf_dim, hp.dec_dropout, device,
                      max_length=hp.max_seq_len)
    model = Transformer(encoder, decoder, device).to(device)

    # Generate
    gen_data = json.load(open(gen_dir, "r"))
    gen_ids = set_gen_ids(gen_data, log.debug)
    if args.debug: gen_ids = gen_ids[:20]  ### DEBUG MODE
    gen_loader = get_loader(protein_data=gen_data, ids=gen_ids,
                            hp=hp, device=device, shuffle=False)
    model = load_model(model_loc, model, device, log)
    model = model.to(device)
    log('Begin generation:')
    new_pssm = generate_whole(model, gen_loader, device, hp, log)  # [N, 700, 22]
    save_pssm(new_pssm, hp, log, filename=output_pssm_file)


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', required=False, default=False)
    main(parser.parse_args())
    print('\nCompelted file: ', os.path.basename(__file__))
