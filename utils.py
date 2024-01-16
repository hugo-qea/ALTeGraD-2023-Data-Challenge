import numpy as np
import scipy.sparse as sp
import networkx as nx
from torch import nn
import torch.nn.functional as F

def normalize_adjacency(A):


    ##################
    # your code here #
    n= A.shape[0]
    A_normalized = A + sp.identity(n)
    degs = np.sum(A_normalized,axis=1).flatten()
    inv_degs = np.power(degs,-0.5).tolist()[0]
    D = sp.diags(inv_degs)
    A_normalized = D.dot(A_normalized).dot(D)
    ##################
    
    return A_normalized


def compute_adjacency(edge_tensor):
    
    adj = np.zeros((edge_tensor.shape[1], edge_tensor.shape[1]))
    for i in range(edge_tensor.shape[1]):
        adj[edge_tensor[0, i], edge_tensor[1, i]] = 1
        adj[edge_tensor[1, i], edge_tensor[0, i]] = 1
    return adj


def generate_walks(edge_tensor, num_walks, walk_length):
    walks = []
    
    ##################
    # your code here #
    for edge in edge_tensor[0,:]:
        for _ in range(num_walks):
            walks.append(random_walk(edge_tensor, edge, walk_length))
    permuted_walks = np.random.permutation(walks)
    ##################

    return permuted_walks.tolist()


def random_walk(edge_tensor, edge, walk_length):

    ##################
    # your code here #
    walk = [node]
    for k in range(walk_length):
        next_node = np.random.choice(list(G.neighbors(walk[k])))
        walk.append(next_node)
    ##################
    
    walk = [str(node) for node in walk]
    return walk