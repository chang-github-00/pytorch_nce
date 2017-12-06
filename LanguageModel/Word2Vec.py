import torch
import torch.nn


class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, factor_matrix=None):
        super(Word2Vec, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = torch.nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = torch.nn.Linear(128, vocab_size)
        self.factor_matrix = factor_matrix

    def forward(self, context_ids, target_ids, noise_ids):
        context_embeddings = self.embeddings(context_ids)
        target_embeddings = self.embeddings(target_ids)
        noise_embeddings = self.embeddings(noise_ids)
        return context_embeddings, target_embeddings, noise_embeddings
