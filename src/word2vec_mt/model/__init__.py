'''
The model for learning the word2vec embeddings is based on the `Embeddings chapter
<https://web.stanford.edu/~jurafsky/slp3/5.pdf>`__ of the `Speech and Language Processing (3rd ed.)
book <https://web.stanford.edu/~jurafsky/slp3/>`__.
It involves creating a sigmoid binary classifier that is trained to classify skipgram context
tokens.
'''

from word2vec_mt.model.tuner import (
    optimise_skipgram_batch_size, tune_skipgram_model, train_best_skipgram_model,
    tune_linear_model, train_best_linear_model,
)
