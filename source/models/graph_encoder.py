from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GAT, VGAE, GraphSAGE, AttentiveFP, MLP
from torch_geometric.nn.conv import GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.norm import GraphNorm


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) module proposed for the Baseline model.

    Args:
        num_node_features (int): Number of input node features.
        nout (int): Number of output node features.
        nhid (int): Number of hidden units.
        graph_hidden_channels (int): Number of hidden channels in the graph convolutional layers.
    """

    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GCN, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        """
        Forward pass of the graph encoder model.

        Args:
            graph_batch (torch_geometric.data.Batch): The input graph batch.

        Returns:
            torch.Tensor: The output tensor after passing through the encoder.
        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
    
class AttentionEncoder(nn.Module):
    """
    AttentionEncoder class implements an attention-based graph encoder.

    Args:
        nout (int): Number of output features.
        nhid (int): Number of hidden features.
        attention_hidden (int): Number of hidden features in the attention mechanism.
        n_in (int): Number of input features.
        dropout (float): Dropout probability.

    Attributes:
        dropout (float): Dropout probability.
        n_in (int): Number of input features.
        attention_hidden (int): Number of hidden features in the attention mechanism.
        n_hidden (int): Number of hidden features.
        n_out (int): Number of output features.
        relu (torch.nn.ReLU): ReLU activation function.
        fc1 (torch.nn.Linear): Linear layer for the first fully connected operation.
        fc2 (torch.nn.Linear): Linear layer for the second fully connected operation.
        Attention (GAT): Graph Attention Network module.

    """

    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(AttentionEncoder, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_out)
        self.Attention = GAT(in_channels=self.n_in, hidden_channels=self.attention_hidden, out_channels=self.n_hidden, dropout=self.dropout, num_layers=2, v2=True, norm=GraphNorm(in_channels = self.n_hidden))

    def forward(self, graph_batch):
        """
        Forward pass of the AttentionEncoder.

        Args:
            graph_batch (torch_geometric.data.Batch): Input graph batch.

        Returns:
            torch.Tensor: Output tensor after passing through the encoder.

        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.Attention(x, edge_index)
        x = self.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
    
class SAGEEncoder(nn.Module):
    """
    SAGEEncoder is a graph encoder module that applies GraphSAGE algorithm to encode graph data.

    Args:
        nout (int): Number of output features.
        nhid (int): Number of hidden units.
        sage_hidden (int): Number of hidden units in the GraphSAGE layers.
        n_in (int): Number of input features.
        dropout (float): Dropout probability.

    Attributes:
        dropout (float): Dropout probability.
        n_in (int): Number of input features.
        sage_hidden (int): Number of hidden units in the GraphSAGE layers.
        n_hidden (int): Number of hidden units.
        n_out (int): Number of output features.
        relu (nn.ReLU): ReLU activation function.
        fc1 (nn.Linear): Fully connected layer 1.
        fc2 (nn.Linear): Fully connected layer 2.
        SAGE (GraphSAGE): GraphSAGE module.

    """

    def __init__(self, nout, nhid, sage_hidden, n_in, dropout):
        super(SAGEEncoder, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.sage_hidden = sage_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.sage_hidden, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_out)
        self.SAGE = GraphSAGE(in_channels=self.n_in, hidden_channels=self.sage_hidden, out_channels=self.n_hidden, dropout=self.dropout, num_layers=4)

    def forward(self, graph_batch):
        """
        Forward pass of the SAGEEncoder module.

        Args:
            graph_batch (torch_geometric.data.Batch): Batch of graph data.

        Returns:
            torch.Tensor: Encoded graph features.

        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.SAGE(x, edge_index)
        x = self.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
class GATConvEncoder(nn.Module):
    """
    Graph Attention Network (GAT) Convolutional Encoder.

    Args:
        nout (int): Number of output features.
        nhid (int): Number of hidden features.
        n_heads (int): Number of attention heads.
        n_in (int): Number of input features.
        dropout (float): Dropout rate.

    Attributes:
        dropout (float): Dropout rate.
        n_in (int): Number of input features.
        n_hidden (int): Number of hidden features.
        n_heads (int): Number of attention heads.
        n_out (int): Number of output features.
        relu (nn.LeakyReLU): LeakyReLU activation function.
        fc1 (nn.Linear): Linear layer for output transformation.
        Attention (GATConv): Graph Attention layer.

    """

    def __init__(self, nout, nhid, n_heads, n_in, dropout):
        super(GATConvEncoder, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.n_hidden = nhid
        self.n_heads = n_heads
        self.n_out = nout
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.n_hidden, self.n_out)
        self.Attention = GATConv(in_channels=self.n_in, out_channels=self.n_hidden, heads=self.n_heads, dropout=self.dropout)

    def forward(self, graph_batch):
            """
            Forward pass of the graph encoder model.

            Args:
                graph_batch (torch_geometric.data.Batch): The input graph batch.

            Returns:
                torch.Tensor: The output tensor after the forward pass.
            """
            x = graph_batch.x
            edge_index = graph_batch.edge_index
            batch = graph_batch.batch
            x = self.Attention(x, edge_index)
            x = self.relu(x)
            x = global_mean_pool(x, batch)
            x = self.fc1(x)
            
            return x
    
class AttentiveFPEncoder(nn.Module):
    """
    AttentiveFPEncoder is a module that encodes graph data using the AttentiveFP algorithm.

    Args:
        nout (int): The number of output features.
        nhid (int): The number of hidden features.
        attention_hidden (int): The number of hidden features in the attention mechanism.
        n_in (int): The number of input features.
        dropout (float): The dropout rate.

    Attributes:
        dropout (float): The dropout rate.
        n_in (int): The number of input features.
        attention_hidden (int): The number of hidden features in the attention mechanism.
        n_hidden (int): The number of hidden features.
        n_out (int): The number of output features.
        relu (nn.ReLU): The ReLU activation function.
        fc1 (nn.Linear): The fully connected layer.
        Attention (AttentiveFP): The AttentiveFP module.

    """

    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(AttentiveFPEncoder, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.n_hidden, self.n_out)
        self.Attention = AttentiveFP(in_channels=self.n_in, hidden_channels=self.attention_hidden, out_channels=self.n_hidden, dropout=self.dropout, num_layers=4, num_timesteps=16, edge_dim=1)

    def forward(self, graph_batch):
        """
        Forward pass of the AttentiveFPEncoder module.

        Args:
            graph_batch (torch_geometric.data.Batch): The input graph batch.

        Returns:
            torch.Tensor: The encoded graph representation.

        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.Attention(x, edge_index)
        x = self.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        #x = self.relu(x)
        
        return x
    
class GATPerso(nn.Module):
    """
    Graph Attention Network (GAT) with personalized modifications.

    Args:
        nout (int): Number of output features.
        nhid (int): Number of hidden features.
        n_heads (int): Number of attention heads.
        n_in (int): Number of input features.
        dropout (float): Dropout rate.

    Attributes:
        dropout (float): Dropout rate.
        n_in (int): Number of input features.
        n_hidden (int): Number of hidden features.
        n_heads (int): Number of attention heads.
        n_out (int): Number of output features.
        relu (nn.LeakyReLU): LeakyReLU activation function.
        fc1 (nn.Linear): Fully connected layer 1.
        conv1 (GATv2Conv): GATv2Conv layer 1.
        conv2 (GATv2Conv): GATv2Conv layer 2.
        fc2 (nn.Linear): Fully connected layer 2.
    """

    def __init__(self, nout, nhid, n_heads, n_in, dropout):
        super(GATPerso, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.n_hidden = nhid
        self.n_heads = n_heads
        self.n_out = nout
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.conv1 = GATv2Conv(in_channels=self.n_in, out_channels=self.n_hidden, heads=self.n_heads, dropout=self.dropout)
        self.conv2 = GATv2Conv(in_channels=self.n_heads * self.n_hidden, out_channels=self.n_hidden, heads=1, dropout=self.dropout)
        self.fc2 = nn.Linear(self.n_hidden, self.n_out)
        
    def forward(self, graph_batch):
            """
            Forward pass of the graph encoder model.

            Args:
                graph_batch (torch_geometric.data.Batch): The input graph batch.

            Returns:
                torch.Tensor: The output tensor after passing through the encoder model.
            """
            x = graph_batch.x
            edge_index = graph_batch.edge_index
            batch = graph_batch.batch
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.conv2(x, edge_index)
            x = self.relu(x)
            x = global_mean_pool(x, batch)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            
            return x
    
class GATwMLP(nn.Module):
    """
    Graph Attention Network (GAT) with Multi-Layer Perceptron (MLP) module.

    Args:
        nout (int): Number of output features.
        nhid (int): Number of hidden features.
        n_heads (int): Number of attention heads.
        n_in (int): Number of input features.
        dropout (float): Dropout rate.

    Attributes:
        dropout (float): Dropout rate.
        n_in (int): Number of input features.
        n_hidden (int): Number of hidden features.
        n_heads (int): Number of attention heads.
        n_out (int): Number of output features.
        relu (nn.LeakyReLU): LeakyReLU activation function.
        outer (MLP): Outer MLP module.
        conv1 (GATv2Conv): First GATv2Conv module.
        conv2 (GATv2Conv): Second GATv2Conv module.
        inner (MLP): Inner MLP module.
    """

    def __init__(self, nout, nhid, n_heads, n_in, dropout):
        super(GATwMLP, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.n_hidden = nhid
        self.n_heads = n_heads
        self.n_out = nout
        self.relu = nn.LeakyReLU()
        self.outer = MLP(in_channels=self.n_hidden, hidden_channels=self.n_hidden, out_channels=self.n_out, num_layers=2)
        self.conv1 = GATv2Conv(in_channels=self.n_hidden, out_channels=self.n_hidden, heads=self.n_heads, dropout=self.dropout)
        self.conv2 = GATv2Conv(in_channels=self.n_heads * self.n_hidden, out_channels=self.n_hidden, heads=1, dropout=self.dropout)
        self.inner = MLP(in_channels=self.n_in, hidden_channels=self.n_hidden, out_channels=self.n_hidden, num_layers=2)
        
    def forward(self, graph_batch):
            """
            Forward pass of the graph encoder model.

            Args:
                graph_batch (torch_geometric.data.Batch): The input graph batch.

            Returns:
                torch.Tensor: The output tensor after passing through the encoder.
            """
            x = graph_batch.x
            edge_index = graph_batch.edge_index
            batch = graph_batch.batch
            x = self.inner(x)
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.conv2(x, edge_index)
            x = self.relu(x)
            x = global_mean_pool(x, batch)
            x = self.outer(x)
            #x = self.relu(x)
            
            return x
        
class Transformer(nn.Module):
    """
    Transformer module.

    Args:
        nout (int): Number of output features.
        nhid (int): Number of hidden features.
        n_heads (int): Number of attention heads.
        n_in (int): Number of input features.
        dropout (float): Dropout rate.

    Attributes:
        dropout (float): Dropout rate.
        n_in (int): Number of input features.
        n_hidden (int): Number of hidden features.
        n_heads (int): Number of attention heads.
        n_out (int): Number of output features.
        relu (nn.LeakyReLU): LeakyReLU activation function.
        fc1 (nn.Linear): Fully connected layer 1.
        conv1 (GATv2Conv): GATv2Conv layer 1.
        conv2 (GATv2Conv): GATv2Conv layer 2.
        fc2 (nn.Linear): Fully connected layer 2.
    """

    def __init__(self, nout, nhid, n_heads, n_in, dropout):
        super(Transformer, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.n_hidden = nhid
        self.n_heads = n_heads
        self.n_out = nout
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.n_in, self.n_hidden)
        self.conv = TransformerConv(in_channels=self.n_hidden, out_channels=self.n_hidden, heads=self.n_heads, dropout=self.dropout)
        self.fc2 = nn.Linear(self.n_hidden * self.n_heads, self.n_out)
        
    def forward(self, graph_batch):
            """
            Forward pass of the graph encoder model.

            Args:
                graph_batch (torch_geometric.data.Batch): The input graph batch.

            Returns:
                torch.Tensor: The output tensor after passing through the encoder.
            """
            x = graph_batch.x
            edge_index = graph_batch.edge_index
            batch = graph_batch.batch
            x = self.fc1(x)
            x = self.relu(x)
            x = self.conv(x, edge_index)
            x = self.relu(x)
            x = global_mean_pool(x, batch)
            x = self.fc2(x)
            
            return x
            
    