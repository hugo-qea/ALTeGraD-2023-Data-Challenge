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
from sklearn.metrics.pairwise import sigmoid_kernel, additive_chi2_kernel, chi2_kernel, laplacian_kernel, polynomial_kernel, rbf_kernel
from sklearn.metrics import label_ranking_average_precision_score

import sys
from tqdm import tqdm
from datetime import datetime
from socket import gethostname
import torchpairwise as pw
from torchmetrics.classification import MultilabelRankingAveragePrecision

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


BCEL = torch.nn.BCEWithLogitsLoss()

def negative_sampling_contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.ones(logits.shape[0], device=v1.device)
  eye = torch.diag_embed(labels).to(v1.device)
  return BCEL(logits, eye) + BCEL(torch.transpose(logits, 0, 1), eye)

def compute_score(graph_embeddings, text_embeddings):
    """
    Computes the score for a batch of predictions.

    Args:
        prediction (torch.Tensor): The predictions.
        batch (torch.Tensor): The batch.

    Returns:
        float: The score.
    """
    graph = graph_embeddings.detach().cpu().numpy()
    text = text_embeddings.detach().cpu().numpy()
    cosine_similarity_ = cosine_similarity(graph, text)
    sigmoid_kernel_ = sigmoid_kernel(graph, text)
    ground_truth = np.ones(graph.shape[0])
    ground_truth_ = np.diag(ground_truth)
    score1 = label_ranking_average_precision_score(ground_truth_, cosine_similarity_)
    score2 = label_ranking_average_precision_score(ground_truth_, sigmoid_kernel_)
    return score1, score2

def scores(x_graph, x_test,reference, batch_size=32, device='cuda'):
    cosine_similarity_ = pw.cosine_similarity(x_graph, x_test)
    sigmoid_kernel_ = pw.sigmoid_kernel(x_graph, x_test)
    scoring = MultilabelRankingAveragePrecision(num_labels=batch_size).to(device)
    cosineScore = scoring(cosine_similarity_, reference)
    sigmoidScore = scoring(sigmoid_kernel_, reference)
    return cosineScore, sigmoidScore


def BatchTripletLoss(x_text,x_graph, margin=0.3,batch_size=32, device='cuda'):
    """
    Computes the triplet loss for a batch of vectors.

    Args:
        v1 (torch.Tensor): The first vector batch.
        v2 (torch.Tensor): The second vector batch.
        v3 (torch.Tensor): The third vector batch.
        margin (float): The margin.

    Returns:
        torch.Tensor: The batch triplet loss value.
    """
    anchor = pw.cosine_similarity(x_text, x_graph)
    ones = torch.ones(batch_size, device=device)
    neg = anchor[torch.randperm(anchor.size()[0])]
    d_neg = torch.sum((anchor - neg) ** 2, dim=1)
    loss = torch.relu(- d_neg + margin)
    return torch.mean(loss)
