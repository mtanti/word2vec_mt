'''
x
'''
import os
import random
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from gensim.models import KeyedVectors


#########################################
def scrape(
) -> list[str]:
    '''
    x
    '''
    browser = webdriver.Firefox()
    browser.get('https://malti.mt/termini/glossarji/')
    assert browser.title == 'Glossarji - Malti'

    (category_select, results_select) = browser.find_elements(
        By.CSS_SELECTOR,
        '.selection [role="combobox"]',
    )

    category_select.click()
    next(
        li
        for li in browser.find_elements(
            By.CSS_SELECTOR,
            '.select2-results li'
        )
        if li.text == 'Oqsma'
    ).click()

    results_select.click()
    next(
        li
        for li in browser.find_elements(
            By.CSS_SELECTOR,
            '.select2-results li'
        )
        if li.text == 'Riżultati'
    ).click()

    while True:
        rows = browser.find_elements(
            By.CSS_SELECTOR,
            '#DataTables_Table_0 .sorting_1',
        )
        if len(rows) > 0:
            break

    results = [row.text for row in rows]

    browser.quit()

    return results


#########################################
def parse(
    rows: list[str],
) -> list[tuple[str, str]]:
    '''
    x
    '''
    results = []
    for row in rows:
        parts = row.split('\nen.')
        mt = parts[0]
        for en in parts[1:]:
            results.append((mt, en))

    return results


#########################################
def filter_results(
    translations: list[tuple[str, str]],
    mt_vocab: set[str],
    en_vocab: set[str],
) -> list[tuple[str, str]]:
    '''
    x
    '''
    results: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for (mt, en) in translations:
        mt = mt.lower()
        en = en.lower()

        if (mt, en) in seen:
            continue

        if ' ' in mt:
            continue
        if mt not in mt_vocab:
            continue

        if en == '':
            continue
        if ' ' in en:
            continue
        if en not in en_vocab:
            continue

        results.append((mt, en))
        seen.add((mt, en))

    return results


#########################################
def main(
) -> None:
    '''
    x
    '''
    print('Getting the mt-en bilingual dictionary')

    print('- Loading the vocabularies')
    with open(os.path.join('data', 'mt_vocab.txt'), 'r', encoding='utf-8') as f:
        mt_vocab = set(f.read().strip().split('\n'))
    en_vocab = set(KeyedVectors.load(os.path.join('data', 'word2vec_en.wordvectors')).key_to_index)

    print('- Scraping malti.mt glossary')
    results = parse(scrape())
    with open(os.path.join('data', 'bilingual_dict_raw.json'), 'w', encoding='utf-8') as f:
        json.dump([
            {'mt': mt, 'en': en}
            for (mt, en) in results
        ], f, ensure_ascii=False, indent=4)

    print('- Creating data set')
    results = filter_results(results, mt_vocab, en_vocab)

    rng = random.Random(0)
    rng.shuffle(results)
    size = len(results)
    train = results[0:int(size*0.6)]
    dev = results[int(size*0.6):int(size*0.8)]
    test = results[int(size*0.8):size]
    with open(os.path.join('data', 'bilingual_dict_train.json'), 'w', encoding='utf-8') as f:
        json.dump([
            {'mt': mt, 'en': en}
            for (mt, en) in train
        ], f, ensure_ascii=False, indent=4)
    with open(os.path.join('data', 'bilingual_dict_dev.json'), 'w', encoding='utf-8') as f:
        json.dump([
            {'mt': mt, 'en': en}
            for (mt, en) in dev
        ], f, ensure_ascii=False, indent=4)
    with open(os.path.join('data', 'bilingual_dict_test.json'), 'w', encoding='utf-8') as f:
        json.dump([
            {'mt': mt, 'en': en}
            for (mt, en) in test
        ], f, ensure_ascii=False, indent=4)

    print('- Done')


#########################################
if __name__ == '__main__':
    main()
