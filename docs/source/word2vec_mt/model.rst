model
=====

The model for learning the word2vec embeddings is based on the `Embeddings chapter
<https://web.stanford.edu/~jurafsky/slp3/5.pdf>`__ of the `Speech and Language Processing (3rd ed.)
book <https://web.stanford.edu/~jurafsky/slp3/>`__.
It involves creating a sigmoid binary classifier that is trained to classify skipgram context
tokens.

.. toctree::
    :maxdepth: 1

    model/data
    model/evaluate
    model/model
    model/trainer
    model/tuner
