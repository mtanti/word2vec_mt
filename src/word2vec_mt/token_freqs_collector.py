'''
Collect the token frequencies of the tokens in the vocabulary in the Maltese corpus.
'''

import collections
import json
import tqdm
from word2vec_mt.paths import vocab_mt_path, corpus_mt_path, freqs_mt_path


#########################################
def collect_token_freqs(
) -> None:
    '''
    Go through the Maltese corpus and collect the frequencies of the tokens in the extracted
    vocabulary.
    '''
    print('Collecting token frequencies from the Maltese corpus')
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

    print('- Collecting token frequencies')
    token_freqs: collections.Counter[int] = collections.Counter()
    with open(corpus_mt_path, 'r', encoding='utf-8') as f:
        for (line, _) in zip(f, tqdm.tqdm(range(num_lines))):
            tokens = [
                token2index[token]
                for token in line.strip().split(' ')
                if token in token2index
            ]
            token_freqs.update(tokens)

    print('- Saving token frequencies')
    with open(freqs_mt_path, 'w', encoding='utf-8') as f:
        json.dump(token_freqs.most_common(), f)

    print('- Done')
