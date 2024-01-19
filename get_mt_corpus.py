'''
x
'''
import os
import collections
import tqdm
import datasets
import tokenise


#########################################
def main(
) -> None:
    '''
    x
    '''
    print('Getting the Maltese corpus')

    print('- Loading the MLRS corpus from HuggingFace')
    dataset = datasets.load_dataset('MLRS/korpus_malti', trust_remote_code=True)

    print('- Tokenising and saving the corpus')
    token_freqs = collections.Counter[str]()
    tokeniser = tokenise.MTWordTokenizer()
    with open(os.path.join('data', 'mt_corpus.txt'), 'w', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(len(dataset['train']))):
            sent_tokens = tokeniser.tokenize(dataset['train'][i]['text'])
            token_freqs.update(sent_tokens)
            print(' '.join(sent_tokens), file=f)

    print('- Saving the vocabulary')
    with open(os.path.join('data', 'mt_vocab.txt'), 'w', encoding='utf-8') as f:
        for (token, freq) in token_freqs.items():
            if freq >= 5:
                print(token, file=f)

    print('- Done')


#########################################
if __name__ == '__main__':
    main()
