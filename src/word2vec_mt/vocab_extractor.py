'''
Extract a vocabulary from the Maltese corpus.
'''

import re
import collections
import tqdm
from word2vec_mt.paths import corpus_mt_path, vocab_mt_path


#########################################
MIN_FREQ_VOCAB = 10
'''
The minimum frequency a token must occur in the corpus to be included in the vocabulary.
'''

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÀÈÌÒÙĊĠĦŻabcdefghijklmnopqrstuvwxyzàáéíóúċġħż'
'''
All the letters of the Maltese alphabet (plus some English ones) as a non-delimited string.
à turns out to be a very common character in the MLRS corpus, so including it.
'''

TOKEN_FILTER_RE = re.compile(f'([-ʼ\']*)([{ALPHABET}]([-ʼ\']*))+')
'''
A regular expression used to filter tokens that we want to keep.
A valid token must have a least one letter of the alphabet and can have dashes (il-) or
apostrophes (ta').
'''


#########################################
def extract_vocab(
) -> None:
    '''
    Extract a vocabulary consisting of all the tokens that occur at least
    MIN_FREQ_VOCAB times.
    '''
    print('Extracting vocabulary')

    print('- Counting lines in corpus')
    num_lines = 0
    with open(corpus_mt_path, 'r', encoding='utf-8') as f:
        for line in f:
            num_lines += 1

    print('- Reading corpus tokens')
    token_freqs = collections.Counter[str]()
    with open(corpus_mt_path, 'r', encoding='utf-8') as f:
        for (line, _) in zip(f, tqdm.tqdm(range(num_lines))):
            sent_tokens = [
                token for token in line.strip().split(' ')
                if TOKEN_FILTER_RE.fullmatch(token)
            ]
            token_freqs.update(sent_tokens)

    print('- Saving the vocabulary')
    with open(vocab_mt_path, 'w', encoding='utf-8') as f:
        for (token, freq) in sorted(
            token_freqs.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if freq >= MIN_FREQ_VOCAB:
                print(token, file=f)

    print('- Done')
