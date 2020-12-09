'''
Main.py
This script trains a transformer to generate PSSMs.
It also generates PSSMs for a specified dataset of protein sequences.
'''
import argparse
import random
import time

from models.transformer import Transformer, Encoder, Decoder
from utils.data_handler import *
from utils.train_utils import *
from utils.utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Prevent Error #15

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Hyperparameters():
    def __init__(self, args):
        # Experiment settings:
        self.n_epochs = 1000
        self.batch_size = 16
        self.num_workers = 4
        self.lr = 1e-5
        self.num_features = 51
        self.early_stop = 10

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

        # Loss setting
        self.mse_loss_reduction = ''  # sum, none

        # Data:
        self.data_dir_tr = 'data/TR5534.json'
        self.data_dir_cb = 'data/CB513.json'  # not used for training
        self.data_dir_no_pssm = 'data/TR6614_no_pssm_onedim.json'
        self.data_file_new_pssm = 'TR6614_transformer_pssm.npy'
        if args.debug:
            self.n_epochs = 2
            self.max_seq_len = 200
            self.data_dir_tr = 'data/CB513.json'


def main(args):
    # Experiment settings
    experiment_name = 'Transformer Baseline'
    log = Log(args.debug)
    hp = Hyperparameters(args)
    device = set_device(log)

    # Record experiment
    debug_warning(args.debug, log)
    log(experiment_name)
    log(vars(hp))

    # Dataloaders
    train_data = json.load(open(hp.data_dir_tr, "r"))

    train_ids, val_ids = splits(train_data, log.debug, SEED)
    train_loader = get_loader(protein_data=train_data, ids=train_ids,
                              hp=hp, device=device, shuffle=True)
    val_loader = get_loader(protein_data=train_data, ids=val_ids,
                            hp=hp, device=device, shuffle=False)

    # Model init
    encoder = Encoder(hp.input_dim, hp.hid_dim, hp.enc_layers, hp.enc_heads, hp.enc_pf_dim, hp.enc_dropout, device,
                      max_length=hp.max_seq_len)
    decoder = Decoder(hp.input_dim, hp.hid_dim, hp.dec_layers, hp.dec_heads, hp.dec_pf_dim, hp.dec_dropout, device,
                      max_length=hp.max_seq_len)
    model = Transformer(encoder, decoder, device).to(device)
    model.apply(initialize_weights)

    # Criterion, optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Trackers
    best_valid_loss = float('inf')
    losses = []
    early_stop_count = hp.early_stop

    # Training
    for epoch in range(hp.n_epochs):
        memory_usage()
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, scheduler, device)
        valid_loss = evaluate(model, val_loader, criterion, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Record epoch
        log(
            f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        losses.append((train_loss, valid_loss))
        graph_losses(losses, log)

        # Track best val, save model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stop_count = hp.early_stop  # reset early stop
            if not log.debug:
                model_dict = {'model': model.state_dict()}
                location = '{}/model_dict.bin'.format(log.main_dir)
                torch.save(model_dict, location)
                log('Saved model.')
                del model_dict
                torch.cuda.empty_cache()
                # save_model(model, optimizer, log)
        else:
            early_stop_count -= 1
            if early_stop_count == 0:
                log('Stopping early! Epoch: ', epoch)
                break
        del train_loss, valid_loss
        torch.cuda.empty_cache()
        ######## END EPOCH ########

    # Generate and save new PSSM
    tr6614 = json.load(open(hp.data_dir_no_pssm, "r"))
    gen_ids = set_gen_ids(tr6614, log.debug)
    gen_loader = get_loader(protein_data=tr6614, ids=gen_ids,
                            hp=hp, device=device, shuffle=False)
    checkpoint = torch.load('{}/model_dict.bin'.format(log.main_dir), map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    new_pssm = generate_whole(model, gen_loader, device, hp, log)  # [N, 700, 22]
    save_pssm(new_pssm, hp, log)
    ######## END MAIN ########


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', required=False, default=False)
    main(parser.parse_args())
    print('\nCompelted file: ', os.path.basename(__file__))
