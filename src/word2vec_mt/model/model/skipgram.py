'''
Classes related to the skipgram model.
'''

import torch
import numpy as np


#########################################
class SkipgramModel(torch.nn.Module):
    '''
    The skipgram model for obtaining Maltese word2vec token vectors.
    '''

    #########################################
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        init_stddev: float,
    ) -> None:
        '''
        The initialiser for creating the uninitialised neural network.

        :param vocab_size: The (Maltese) vocabulary size.
        :param embedding_size: The vector size for the (Maltese) token vectors.
        :param init_stddev: The standard deviation to use when initialising the parameters using a
            random normal distribution.
        '''
        super().__init__()
        self.init_stddev = init_stddev
        self.target_embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)
        self.context_embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)

    #########################################
    def initialise(
        self,
        seed: int,
    ) -> None:
        '''
        Initialise the model parameters.

        :param seed: The random seed to use.
        '''
        g = torch.Generator()
        g.manual_seed(seed)
        for (_, param) in self.named_parameters():
            torch.nn.init.normal_(param, std=self.init_stddev, generator=g)

    #########################################
    def forward(
        self,
        target_indexes: torch.Tensor,
        context_indexes: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Get the dot-product between a set of target tokens and their corresponding context tokens.

        :param target_indexes: The vector of token indexes of the target tokens.
        :param context_indexes: The vector of token indexes of the context tokens.
        :return: The vector of dot-products.
        '''
        target_embedded = self.target_embedding_layer(target_indexes)
        context_embedded = self.context_embedding_layer(context_indexes)
        return (target_embedded*context_embedded).sum(dim=1)

    #########################################
    def get_embeddings(
        self,
    ) -> np.ndarray:
        '''
        Get a NumPy array of the word2vec embedding matrix (where token vectors are the row-vectors
        which are in vocabulary order).

        :return: The embedding matrix.
        '''
        return self.target_embedding_layer.weight.data.detach().cpu().numpy()
