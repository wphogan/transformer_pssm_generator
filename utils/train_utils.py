import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.utils import unnormalize, trim_padding, memory_usage


def train(model, iterator, optimizer, criterion, scheduler, device):
    model.train()
    epoch_loss = 0
    for step, batch in enumerate(iterator):
        src, trg, lens = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        src, trg = add_sos_src(src, device), add_sos_trg(trg, device)
        optimizer.zero_grad()
        output, _ = model(src, trg, lens)
        output, trg = trim_padding(output, trg, lens, device)  # Trim padding

        # Loss & gradient step
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg, lens = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            src, trg = add_sos_src(src, device), add_sos_trg(trg, device)
            output, _ = model(src, trg, lens)
            output, trg = trim_padding(output, trg, lens, device)
            loss = criterion(output.to(device), trg.to(device))
            epoch_loss += loss.item()
        del model, output, trg, src, loss, _, batch, lens
        torch.cuda.empty_cache()
    return epoch_loss / len(iterator)


def generate_whole(model, iterator, device, hp, log):
    model.eval()
    new_pssm = np.zeros((1, hp.max_seq_len + 1, 22))
    with torch.no_grad():
        for batch in tqdm(iterator, leave=False):
            src, trg, lens = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            src, trg = add_sos_src(src, device), add_sos_trg(trg, device)
            output, _ = model(src, trg, lens)
            output = unnormalize(output, device)
            new_pssm = np.concatenate((new_pssm, output.cpu().numpy()), axis=0)
    log('Completed whole generation.')
    new_pssm = new_pssm[1:, 1:, :]
    log(new_pssm.shape)
    return new_pssm


def generate(model, iterator, device, hp, log):
    model.eval()
    new_pssm = np.zeros((1, hp.max_seq_len + 1, 22))
    with torch.no_grad():
        for batch in tqdm(iterator, leave=False):
            memory_usage()
            log('.')
            src, trg, lens = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            bs = src.shape[0]
            src_mask = model.make_src_mask(src, lens).to(device)
            enc_src = model.encoder(src, src_mask).to(device)
            targets = torch.zeros((bs, 1, 22), dtype=torch.long).to(device)  # sos
            for i in range(hp.max_seq_len):
                trg_tensor = targets.float().to(device)
                trg_lens = torch.ones([bs, 1], dtype=torch.long, device=device) * (i + 1)  # Lens for trg
                trg_mask = model.make_trg_mask(trg_tensor, trg_lens).to(device)
                with torch.no_grad():
                    output, _ = model.decoder(trg_tensor.float(), enc_src, trg_mask, src_mask)
                output = unnormalize(output, device)
                pred = output[:, -1, :].unsqueeze(1)
                targets = torch.cat((targets.float().to(device), pred.to(device)), dim=1)  # append prediction
            new_pssm = np.concatenate((new_pssm, targets.cpu().numpy()), axis=0)
            del targets, pred, output, _, trg, src, lens, src_mask, enc_src, trg_tensor, trg_lens, trg_mask, bs
            torch.cuda.empty_cache()
    log('Completed generation.')
    new_pssm = new_pssm[1:, 1:, :]  # remove first batch and sos
    return new_pssm


def add_sos_src(src, device):
    bs = src.shape[0]
    sos = torch.zeros((bs, 1), dtype=torch.long).to(device)
    src = torch.cat((sos, src), dim=1)
    return src.to(device)


def add_sos_trg(trg, device):
    bs = trg.shape[0]
    sos = torch.zeros((bs, 1, 22), dtype=torch.float).to(device)
    trg = torch.cat((sos, trg), dim=1)
    return trg.to(device)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model, log):
    log(f'The model has {count_parameters(model):,} trainable parameters')
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
