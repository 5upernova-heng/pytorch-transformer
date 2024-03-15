import os

import numpy as np
import torch

dir_ = "./data"


def save_tensor(tensor: torch.Tensor, filename: str):
    if not os.path.exists(dir_):
        os.mkdir(dir_)

    t = tensor.detach().numpy()[0]
    np.savetxt(f"{dir_}/{filename}", t, header=str(t.shape))
