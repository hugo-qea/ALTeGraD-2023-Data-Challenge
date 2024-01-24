from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GAT, VGAE, GraphSAGE, AttentiveFP, MLP
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool
from transformers import AutoModel


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
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
    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(AttentionEncoder, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_out)
        self.Attention = GAT(in_channels=self.n_in, hidden_channels = self.attention_hidden, out_channels=self.n_hidden, dropout=self.dropout, num_layers=4, v2=True)

    def forward(self, graph_batch):
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
        self.SAGE = GraphSAGE(in_channels=self.n_in, hidden_channels = self.sage_hidden, out_channels=self.n_hidden, dropout=self.dropout, num_layers=4)
    def forward(self, graph_batch):
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
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.Attention(x, edge_index)
        x = self.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        #x = self.relu(x)
        
        return x
    
class AttentiveFPEncoder(nn.Module):
    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(AttentiveFPEncoder, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.n_hidden, self.n_out)
        self.Attention = AttentiveFP(in_channels=self.n_in,hidden_channels=self.attention_hidden, out_channels=self.n_hidden, dropout=self.dropout,num_layers=4, num_timesteps=16, edge_dim=1)
    def forward(self, graph_batch):
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
        self.inner = MLP(in_channels=self.n_in, hidden_channels=self.n_hidden,out_channels=self.n_hidden,  num_layers=2)
        
    def forward(self, graph_batch):
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
        
    

    
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:,0,:]
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    
class ModelAttention(nn.Module):
    def __init__(self, model_name, n_in, nout, nhid, attention_hidden, dropout):
        super(ModelAttention, self).__init__()
        self.graph_encoder = AttentionEncoder(nout, nhid, attention_hidden, n_in, dropout)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    
    
    
class ModelSAGE(nn.Module):
    def __init__(self, model_name, n_in, nout, nhid, sage_hidden, dropout):
        super(ModelSAGE, self).__init__()
        self.graph_encoder = SAGEEncoder(nout, nhid, sage_hidden, n_in, dropout)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    
class ModelGATConv(nn.Module):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelGATConv, self).__init__()
        self.graph_encoder = GATConvEncoder(nout, nhid, n_heads, n_in, dropout)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

class ModelAttentiveFP(nn.Module):
    def __init__(self, model_name, n_in, nout, nhid, attention_hidden, dropout):
        super(ModelAttentiveFP, self).__init__()
        self.graph_encoder = AttentiveFPEncoder(nout, nhid, attention_hidden, n_in, dropout)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    
class ModelGATPerso(nn.Module):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelGATPerso, self).__init__()
        self.graph_encoder = GATPerso(nout, nhid, n_heads, n_in, dropout)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    

class ModelGATwMLP(nn.Module):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelGATwMLP, self).__init__()
        self.graph_encoder = GATwMLP(nout, nhid, n_heads, n_in, dropout)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder