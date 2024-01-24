from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader, Data
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as TorchGeoDataLoader
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
from torch_geometric.nn import summary
import time
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys
from tqdm import tqdm
from datetime import datetime
from socket import gethostname

CE = torch.nn.CrossEntropyLoss()



def contrastive_loss(v1, v2):
    """
    Computes the contrastive loss between two vectors.

    Args:
        v1 (torch.Tensor): The first vector.
        v2 (torch.Tensor): The second vector.

    Returns:
        torch.Tensor: The contrastive loss value.
    """
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)
