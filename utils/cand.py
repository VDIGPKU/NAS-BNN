import torch


def cand2tuple(cand):
    return tuple(cand.int().flatten().tolist())


def tuple2cand(cand_tuple):
    return torch.tensor(cand_tuple).reshape(-1, 6)
