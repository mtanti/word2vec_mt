'''
Classes related to the linear model.
'''

import torch


#########################################
class LinearModel(torch.nn.Module):
    '''
    The linear model for translating Maltese token vectors into English token vectors.
    '''

    #########################################
    def __init__(
        self,
        source_embedding_size: int,
        target_embedding_size: int,
        init_stddev: float,
        use_bias: bool,
    ) -> None:
        '''
        The initialiser for creating the uninitialised neural network.

        :param source_embedding_size: The vector size for the source tokens (Maltese).
        :param target_embedding_size: The vector size for the target tokens (English).
        :param init_stddev: The standard deviation to use when initialising the parameters using a
            random normal distribution.
        :param use_bias: Whether to use a bias in the linear model.
        '''
        super().__init__()
        self.init_stddev = init_stddev
        self.layer = torch.nn.Linear(source_embedding_size, target_embedding_size, bias=use_bias)

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
        source_vectors: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Translate the source vectors.

        :param source_vectors: The matrix of source row-vectors.
        :return: The matrix of target row-vectors.
        '''
        return self.layer(source_vectors)
