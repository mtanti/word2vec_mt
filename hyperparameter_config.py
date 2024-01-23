'''
x
'''
import pydantic

# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec

class SearchSpace(pydantic.BaseModel):
    '''
    x
    '''
    gensim_window: list[int]
    gensim_sg: list[int]
    gensim_negative: list[int]
    gensim_sample: list[float]
    gensim_epochs: list[int]
    sklearn_fit_intercept: list[bool]
    sklearn_alpha: list[float]


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


class HyperparameterSearchConfig(pydantic.BaseModel):
    '''
    x
    '''
    num_iters: int
    space: SearchSpace
