from gensim.models import Word2Vec
from utils import *


class DeepWalk:
    def __init__(self, G, num_walks, walk_length, n_dim):
        self.G = G
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.n_dim = n_dim

    def train(self):
        #print("Generating walks")
        walks = generate_walks(self.G, self.num_walks, self.walk_length)

        #print("Training word2vec")
        model = Word2Vec(vector_size=self.n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
        model.build_vocab(walks)
        model.train(walks, total_examples=model.corpus_count, epochs=5)

        return model