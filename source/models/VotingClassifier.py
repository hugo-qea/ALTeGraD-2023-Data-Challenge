from models.graph_encoder import *
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch


    
class TextEncoder(nn.Module):
    def __init__(self, model_name,n_input=768,n_output=300):
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
            self.graph_encoder1 = None
            self.graph_encoder2 = None
            self.graph_encoder3 = None
            self.graph_encoder4 = None
            self.text_encoder = TextEncoder(model_name)
            
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
            graph_encoded1 = self.graph_encoder1(graph_batch)
            graph_encoded2 = self.graph_encoder2(graph_batch)
            graph_encoded3 = self.graph_encoder3(graph_batch)
            graph_encoded4 = self.graph_encoder4(graph_batch)
            text_encoded = self.text_encoder(input_ids, attention_mask)
            return graph_encoded1, graph_encoded2, graph_encoded3, graph_encoded4, text_encoded
        
        def get_text_encoder(self):
            """
            Returns the text encoder of the model.

            Returns:
                text_encoder: The text encoder.

            """
            return self.text_encoder
        
        def get_graph_encoder1(self):
            """
            Returns the graph encoder of the model.

            Returns:
                graph_encoder1: The first graph encoder.

            """
            return self.graph_encoder1
        
        def get_graph_encoder2(self):
            
            return self.graph_encoder2
        
        def get_graph_encoder3(self):
            
            return self.graph_encoder3
        
        def get_graph_encoder4(self):
            
            return self.graph_encoder4



        
        def get_model_surname(self):
            """
            Gets the surname of the model.

            Returns:
                model_surname: The surname of the model.

            """
            pass


class VotingClassifier(Model):
	def __init__(self, model_name, graph_encoder1, graph_encoder2, graph_encoder3, graph_encoder4):
		"""
		Initializes a VotingClassifier object.

		Args:
			model_name (str): The name of the pretrained text encoder.
			graph_encoder1: The first graph encoder.
			graph_encoder2: The second graph encoder.
			graph_encoder3: The third graph encoder.
			graph_encoder4: The fourth graph encoder.

		"""
		super(VotingClassifier, self).__init__(model_name)
		self.graph_encoder1 = graph_encoder1
		self.graph_encoder2 = graph_encoder2
		self.graph_encoder3 = graph_encoder3
		self.graph_encoder4 = graph_encoder4
		
	
	def get_model_surname(self):
		"""
		Gets the surname of the model.

		Returns:
			model_surname: The surname of the model.

		"""
		return "VotingClassifier"
