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
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score

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

def nce_loss(v1, v2, temperature=1.0, num_neg_samples=10):
    """
    Computes the Noise-Contrastive Estimation (NCE) loss between two vectors.

    Args:
        v1 (torch.Tensor): The first vector.
        v2 (torch.Tensor): The second vector.
        temperature (float): A scaling factor for the logits.
        num_neg_samples (int): Number of negative samples.

    Returns:
        torch.Tensor: The NCE loss value.
    """
    batch_size = v1.shape[0]

    # Sample negative indices
    noise_indices = torch.randint(high=batch_size, size=(batch_size, num_neg_samples), device=v1.device)

    # Concatenate positive target indices with negative indices
    all_indices = torch.cat([torch.arange(batch_size, device=v1.device).unsqueeze(1), noise_indices], dim=1)

    # Get embeddings for positive and negative samples
    embeddings = torch.cat([v1.unsqueeze(1), v2[noise_indices]], dim=1)

    # Calculate the logits
    logits = torch.matmul(v1, v2.t()) / temperature

    # Create labels for positive and negative samples
    labels = torch.zeros(batch_size, dtype=torch.long, device=v1.device)

    # Compute NCE loss
    nce_loss = F.cross_entropy(logits, labels)

    return nce_loss


#def snn_loss()
