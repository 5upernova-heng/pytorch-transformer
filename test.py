"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import os
import time

import numpy as np

from torch import nn
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        m.weight.data.fill_(0)
        print(m.weight)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)



def save(tensor: torch.Tensor, filename: str):
    t = tensor.detach().numpy()[0]
    np.savetxt(filename, t, header=str(t.shape))

if __name__ == '__main__':
    model.train()
    for batch in train_iter:
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        start_time = time.time()
        output = model(src, trg[:, :-1])
        print("One step complete. %.2fs" % (time.time() - start_time))
        dir_ = "./data"
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        save(src, f"{dir_}/src.txt")
        save(trg, f"{dir_}/trg.txt")
        save(output, f"{dir_}/output.txt")
        break
