from models.graph_encoder import *
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch


    
class TextEncoder(nn.Module):
    def __init__(self, model_name,n_input=768,n_output=768):
        """
        Initializes a TextEncoder object.

        Args:
            model_name (str): The name or path of the pre-trained model to be used.

        """
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.train()
        self.linear = nn.Linear(n_input,n_output)
        
    def forward(self, input_ids, attention_mask):
        """
        Performs forward pass of the TextEncoder.

        Args:
            input_ids (torch.Tensor): The input tensor containing the tokenized text.
            attention_mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The encoded text tensor.

        """
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return self.linear(encoded_text.last_hidden_state[:,0,:])
    
class Model(nn.Module):
        def __init__(self, model_name):
            """
            Initializes a Model object.

            Args:
                model_name (str): The name of the pretrained text encoder.

            """
            super(Model, self).__init__()
            self.graph_encoder = None
            self.text_encoder = TextEncoder(model_name)
            self.temp = nn.Parameter(torch.tensor(1.0))
            self.register_parameter('temp', self.temp)
            
        def forward(self, graph_batch, input_ids, attention_mask):
            """
            Performs a forward pass through the model.

            Args:
                graph_batch: The input graph batch.
                input_ids: The input text IDs.
                attention_mask: The attention mask for the input text.

            Returns:
                graph_encoded: The encoded graph.
                text_encoded: The encoded text.

            """
            graph_encoded = self.graph_encoder(graph_batch)
            text_encoded = self.text_encoder(input_ids, attention_mask)
            graph_encoded = graph_encoded * torch.exp(self.temp)
            text_encoded = text_encoded * torch.exp(self.temp)
            return graph_encoded, text_encoded
        
        def get_text_encoder(self):
            """
            Returns the text encoder of the model.

            Returns:
                text_encoder: The text encoder.

            """
            return self.text_encoder
        
        def get_graph_encoder(self):
            """
            Returns the graph encoder of the model.

            Returns:
                graph_encoder: The graph encoder.

            """
            return self.graph_encoder
        
        def get_model_surname(self):
            """
            Gets the surname of the model.

            Returns:
                model_surname: The surname of the model.

            """
            pass
        
        def predict(self, graph_batch, input_ids, attention_mask):
            """
            Predicts the embedding using cosine similarity.

            Args:
                graph_batch: The input graph batch.
                input_ids: The input text IDs.
                attention_mask: The attention mask for the input text.

            Returns:
                similarity: The cosine similarity between the graph and text embeddings.

            """
            graph_encoded, text_encoded = self.forward(graph_batch, input_ids, attention_mask)
            similarity = F.cosine_similarity(graph_encoded, text_encoded)
            return similarity


    
class Baseline(Model):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        """
        Initializes a Baseline model.

        Args:
            model_name (str): The name of the model.
            num_node_features (int): The number of node features.
            nout (int): The number of output features.
            nhid (int): The number of hidden units.
            graph_hidden_channels (int): The number of hidden channels in the graph encoder.
        """
        super(Baseline, self).__init__(model_name)
        self.graph_encoder = GCN(num_node_features, nout, nhid, graph_hidden_channels)
    
    def get_model_surname(self):
        """
        Returns the surname of the model.

        Returns:
            str: The surname of the model.
        """
        return 'Baseline'
        
    
class ModelAttention(Model):
    def __init__(self, model_name, n_in, nout, nhid, attention_hidden, dropout):
        """
        Initialize the ModelAttention class.

        Args:
            model_name (str): The name of the model.
            n_in (int): The number of input features.
            nout (int): The number of output features.
            nhid (int): The number of hidden units.
            attention_hidden (int): The number of hidden units in the attention mechanism.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super(ModelAttention, self).__init__(model_name)
        self.graph_encoder = AttentionEncoder(nout, nhid, attention_hidden, n_in, dropout)
        
    def get_model_surname(self):
        """
        Get the surname of the model.

        Returns:
            str: The surname of the model.
        """
        return 'GATv2'

    
    
    
class ModelSAGE(Model):
    def __init__(self, model_name, n_in, nout, nhid, sage_hidden, dropout):
        super(ModelSAGE, self).__init__(model_name)
        self.graph_encoder = SAGEEncoder(nout, nhid, sage_hidden, n_in, dropout)

    def get_model_surname(self):
        return 'SAGE'
    
class ModelGATConv(Model):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelGATConv, self).__init__(model_name)
        self.graph_encoder = GATConvEncoder(nout, nhid, n_heads, n_in, dropout)

    def get_model_surname(self):
        return 'GATv2Conv'

class ModelAttentiveFP(Model):
    def __init__(self, model_name, n_in, nout, nhid, attention_hidden, dropout):
        super(ModelAttentiveFP, self).__init__(model_name)
        self.graph_encoder = AttentiveFPEncoder(nout, nhid, attention_hidden, n_in, dropout)

    def get_model_surname(self):
        return 'AttentiveFP'
    
class ModelGATPerso(Model):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelGATPerso, self).__init__(model_name=model_name)
        self.graph_encoder = GATPerso(nout, nhid, n_heads, n_in, dropout)

    def get_model_surname(self):
        return 'GATv2Perso'   

class ModelGATwMLP(Model):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelGATwMLP, self).__init__(model_name=model_name)
        self.graph_encoder = GATwMLP(nout, nhid, n_heads, n_in, dropout)

    def get_model_surname(self):
        return 'GATv2wMLP'
    
class ModelTransformer(Model):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelTransformer, self).__init__(model_name=model_name)
        self.graph_encoder = Transformer(nout, nhid, n_heads, n_in, dropout)

    def get_model_surname(self):
        return 'Transformer'
    
class ModelGPS(Model):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelGPS, self).__init__(model_name=model_name)
        self.graph_encoder = GPS(nout, nhid, n_heads, n_in, dropout)

    def get_model_surname(self):
        return 'GPS'
    
class ModelSuperGAT(Model):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelSuperGAT, self).__init__(model_name=model_name)
        self.graph_encoder = SuperGAT(nout, nhid, n_heads, n_in, dropout)

    def get_model_surname(self):
        return 'SuperGAT'
    
class ModelVGAE(Model):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelVGAE, self).__init__(model_name=model_name)
        self.graph_encoder = VGAEncoder(nout, nhid,n_in, dropout)

    def get_model_surname(self):
        return 'VGAE'
    
class ModelGINE(Model):
	def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
		super(ModelGINE, self).__init__(model_name=model_name)
		self.graph_encoder = SuperGIN(nout, nhid,n_heads, n_in, dropout)

	def get_model_surname(self):
		return 'GINE'

class ModelTransformerv2(Model):
    def __init__(self, model_name, n_in, nout, nhid, n_heads, dropout):
        super(ModelTransformerv2, self).__init__(model_name=model_name)
        self.graph_encoder = GraphTransformer(nout, nhid, n_heads, n_in, dropout)

    def get_model_surname(self):
        return 'Transformer_v2'
