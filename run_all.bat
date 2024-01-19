@echo off

call python get_en_word2vec.py || pause && exit /b
call python get_mt_corpus.py || pause && exit /b
call python get_bilingual_dict.py || pause && exit /b
call python tune_mt_word2vec.py || pause && exit /b
