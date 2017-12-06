import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from LanguageModel.NGram import NGramLanguageModeler

logger = logging.Logger(name="dev")
torch.manual_seed(1)

######################################################################

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
VOCAB_SIZE = len(word_to_ix)

losses = []

# Define the loss function
loss_function = nn.NLLLoss()

# Define the language model
model = NGramLanguageModeler(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_SIZE)
model.cuda()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.5)

# Learning variables
batch_size = 16
num_noise_samples = 64
epochs = 1

for epoch in range(epochs):
    total_loss = torch.Tensor([0]).cuda()
    for context, target in trigrams:
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context] # Get all the context word ids
        context_var = autograd.Variable(torch.LongTensor(context_idxs)).cuda()

        # Step 1.b Get all the noise word ids (build a uniform random sampler here)
        # Convert them into autograd Variable
        noise_idxs = np.random.choice(VOCAB_SIZE, num_noise_samples)
        noise_var = autograd.Variable(torch.LongTensor(noise_idxs)).cuda()

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words. This should produce a
        log_probs = model(context_var)

        # Step 4a. Prepare the target idx and then wrap them as a variable
        target_idxs = [word_to_ix[target]]
        target_var = autograd.Variable(torch.LongTensor(target_idxs)).cuda()

        # Step 4b. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, target_var)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Step 6. Update the total loss
        total_loss += loss.data
        break

    losses.append(total_loss)