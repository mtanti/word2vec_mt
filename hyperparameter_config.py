'''
x
'''
import pydantic

class SearchSpace:
    '''
    https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    '''

    gensim_window = [
        5,
        10,
    ]

    gensim_sg = [
        0,
        1,
    ]

    gensim_negative = [
        5,
        10,
        15,
        20,
    ]

    gensim_sample = [
        0.0,
        1e-1,
        1e-2,
        1e-3,
        1e-4,
        1e-5,
    ]

    gensim_epochs = [
        2,
        5,
    ]

    sklearn_fit_intercept = [
        True,
        False,
    ]

    sklearn_alpha = [
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1e0,
    ]

class Hyperparameters(pydantic.BaseModel):
    '''
    x
    '''
    gensim_window: int
    gensim_sg: int
    gensim_negative: int
    gensim_sample: float
    gensim_epochs: int
    sklearn_fit_intercept: bool
    sklearn_alpha: float
