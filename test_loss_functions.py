
import torch
import transformers
from transformers import BertTokenizer
from torch import nn
from loguru import logger
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
import os

def test_l1_loss():
    #
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#L1Loss
    #
    loss = nn.L1Loss()
    inp_arr = [1, 22, 3]
    out_arr = [2, 3, 41]
    inp = torch.tensor(inp_arr, dtype=torch.float)
    target = torch.tensor(out_arr, dtype=torch.float)
    output = loss(inp, target)
    print(output)
    print(np.mean(np.abs(np.array(inp_arr) - np.array(out_arr))))

def test_cross_entropy_loss():
    loss = nn.CrossEntropyLoss()
    inp = torch.tensor([
        [0, 1.0, 0, 0, 0.0],
        [0, 1.0, 0, 0, 0.0],
        [0, 0.0, 0, 0, 1.0],
    ])
    target = torch.tensor([1, 1, 4])  # for 5 - doesn't work
    logger.info(f"CrossEntropyLoss = {loss(inp, target)}")
    output = torch.cat([torch.tensor([0., 1., 0.]).unsqueeze(0),
                        torch.tensor([9., 1., 1.]).unsqueeze(0)],
                       0)
    loss_val = loss(output, torch.tensor([0, 1]))
    logger.info(f"CrossEntropyLoss = {loss_val}")


def test_bce_with_logits_loss():
    loss_bce = nn.BCEWithLogitsLoss()
    inp = torch.tensor([
        [0, 1.0, 0, 0, 0.0],
        [0, 1.0, 0, 0, 0.0],
        [0, 0.0, 0, 0, 1.0],
    ])
    target = inp
    loss_val = loss_bce(inp, target)
    logger.info(f"BCE loss = {loss_val}")
