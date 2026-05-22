'''
Convert the Maltese corpus into a word2vec skip-gram model train set.
'''

from typing import Iterable, Iterator
import tqdm
from word2vec_mt.paths import vocab_mt_path, corpus_mt_path, proccorpus_mt_path
from word2vec_mt.corpus_preprocessor.buffer import BufferedHDF5


#########################################
def _get_words_and_contexts(
    corpus_sents: Iterable[list[str]],
    context_radius: int,
    token2index: dict[str, int],
) -> Iterator[tuple[int, int]]:
    '''
    Get all the words-in-context and context word pairs in a corpus.

    :param corpus_sents: The tokenised sentences of the corpus.
    :param context_radius: The context window radius to use.
    :param token2index: The token-to-index vocabulary mapper.
    :return: A pair of indexes ``(a, b)`` where ``a`` is the word-in-context index
        and ``b`` is the context word index.
    '''
    for sent in corpus_sents:
        sent_len = len(sent)
        for i in range(sent_len - 1):
            token1_index = token2index.get(sent[i])
            if token1_index is None:
                continue
            for j in range(i + 1, min(i + context_radius + 1, sent_len)):
                token2_index = token2index.get(sent[j])
                if token2_index is None:
                    continue
                yield (token1_index, token2_index)
                yield (token2_index, token1_index)


#########################################
def preprocess_corpus(
    context_radius: int,
    buffer_size: int,
) -> None:
    r'''
    Process the Maltese corpus into a training set for a word2vec skip-gram
    model.
    Data is stored in a HDF5 file with data sets called 'radius\_' followed by
    the context window radius used.
    Each data set consists of a 2-colomn int64 matrix with a row for every
    word-in-context / context word pair and where the first column is the index
    of the word-in-context and the second column is the index of the context
    word.

    :param context_radius: The radius of the context window.
    :param buffer_size: The number of rows to buffer in memory before flushing
        into the file.
    '''
    if context_radius <= 0:
        raise ValueError('context_radius must be a positive number.')
    if buffer_size <= 0:
        raise ValueError('buffer_size must be a positive number.')

    print('Preprocessing Maltese corpus into a training set')
    print()

    print('- Loading vocabulary')
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        token2index = {
            line.strip(): index
            for (index, line) in enumerate(f)
        }

    print('- Counting lines in corpus')
    num_lines = 0
    with open(corpus_mt_path, 'r', encoding='utf-8') as f:
        for line in f:
            num_lines += 1

    def get_corpus_sents(
        show_progbar: bool,
    ) -> Iterator[list[str]]:
        with open(corpus_mt_path, 'r', encoding='utf-8') as f:
            if show_progbar:
                for (line, _) in zip(f, tqdm.tqdm(range(num_lines))):
                    yield line.strip().split(' ')
            else:
                for line in f:
                    yield line.strip().split(' ')

    def get_words_and_contexts(
        show_progbar: bool,
    ) -> Iterator[tuple[int, int]]:
        yield from _get_words_and_contexts(
            get_corpus_sents(show_progbar),
            context_radius,
            token2index,
        )

    print('- Counting number of data items')
    num_data_items = 0
    for _ in get_words_and_contexts(True):
        num_data_items += 1

    print('- Saving data items')
    with BufferedHDF5(
        file_path=proccorpus_mt_path,
        buffer_max_size=buffer_size,
        data_set_name=f'radius_{context_radius}',
        num_data_items=num_data_items,
    ).get_handle() as f_out:
        for ((wic, cw), _) in zip(
            get_words_and_contexts(False),
            tqdm.tqdm(range(num_data_items))
        ):
            f_out.append(wic, cw)

    print('- Done')
