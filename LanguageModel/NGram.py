import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(name="dev")


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, contexts):
        # Get all the embedding of the context words
        context_embeddings = self.embeddings(contexts)

        # Convert the context embeddings into 1 long vector
        context_embeddings = context_embeddings.view((1, -1))

        # Run through the neural networks
        out = F.relu(self.linear1(context_embeddings))
        logger.warning(["NGram.out: ", out.size()])

        out = self.linear2(out)
        logger.warning(["NGram.out: ", out.size()])
        log_probs = F.log_softmax(out)
        logger.warning(["NGram.log_probs: ", log_probs.size()])
        return log_probs