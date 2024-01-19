'''
x
'''
import os
import gensim.downloader


#########################################
def main(
) -> None:
    '''
    x
    '''
    print('Getting the English word2vec')
    wordvecs_en = gensim.downloader.load('word2vec-google-news-300')
    wordvecs_en.save(os.path.join('data', 'word2vec_en.wordvectors'))
    print('- Done')


#########################################
if __name__ == '__main__':
    main()
