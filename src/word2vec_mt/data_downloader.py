'''
Download the Maltese and English data sources to be used for creating word2vec
embeddings.
'''
import os
import gzip
import tempfile
import struct
import tqdm
import datasets
import malti.tokeniser
import gdown
import numpy as np
from word2vec_mt.paths import (
    corpus_mt_path, vocab_en_path, word2vec_en_path,
)


#########################################
def download_mt(
) -> None:
    '''
    Download the Maltese corpus from HuggingFace.
    '''
    print('Getting the Maltese corpus')

    print('- Loading the MLRS corpus from HuggingFace')
    dataset = datasets.load_dataset(
        'MLRS/korpus_malti',
        revision='0d61e8e55c55e5397783a26e8ff3b7b4a9360bd6',
        trust_remote_code=True,
    )

    print('- Tokenising and saving the corpus')
    tokeniser = malti.tokeniser.KMTokeniser()
    with open(corpus_mt_path, 'w', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(len(dataset['train']))):
            sent_tokens = tokeniser.tokenise(dataset['train'][i]['text'])
            print(' '.join(sent_tokens), file=f)

    print('- Done')


#########################################
def download_en(
) -> None:
    '''
    Download the English word2vec from Google Code.
    Code for parsing the binary file was taken from:
    https://github.com/danielfrg/word2vec/blob/main/word2vec/wordvectors.py
    '''
    print('Getting the English word2vec')

    with tempfile.TemporaryDirectory() as tmp_dir:
        print('- Downloading GoogleNews word2vec')
        gdown.download(
            'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view',
            os.path.join(tmp_dir, 'GoogleNews-vectors-negative300.bin.gz'),
            fuzzy=True,
        )
        print('- Extracting word2vec')
        with gzip.open(
            os.path.join(tmp_dir, 'GoogleNews-vectors-negative300.bin.gz'),
            'rb',
        ) as f:
            headers = f.readline().split()
            vocab_size = int(headers[0])
            vector_size = int(headers[1])

            vocab: list[str] = []
            vectors = np.empty((vocab_size, vector_size), dtype=np.float32)
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for i in tqdm.tqdm(range(vocab_size)):
                # Read word.
                word_bytes: list[bytes] = []
                while True:
                    byte = f.read(1)
                    if byte == b' ':
                        break
                    word_bytes.append(byte)
                vocab.append(b''.join(word_bytes).decode('utf-8'))

                # Read vector.
                vectors[i] = np.array(
                    struct.unpack(
                        '<'+('f'*vector_size),
                        f.read(binary_len)
                    ),
                    np.float32,
                )

    print('- Saving word2vec')
    with open(vocab_en_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            print(word, file=f)
    np.save(word2vec_en_path, vectors, allow_pickle=False)

    print('- Done')
