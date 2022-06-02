import numpy as np
from gender_novels.corpus import Corpus


def get_corpus():
    print("Loading corpus")
    corpus = Corpus('sample_novels')
    female_corpus = corpus.filter_by_gender('female')
    male_corpus = corpus.filter_by_gender('male')

    return female_corpus, male_corpus


def get_cooccurence_matrix():
    female_corpus, male_corpus = get_corpus()

    # Get novel text (as list of words) and file title
    all_female_novel_text = []
    all_female_novel_name = []
    for novel in female_corpus.novels:
        all_female_novel_name.append(novel.filename)
        all_female_novel_text.extend(novel.get_tokenized_text())
    
    # Get unique words from all novels
    unique_words = list(set(all_female_novel_text))
    num_unique = len(unique_words)

    cooc_mat = np.zeros((num_unique, num_unique),np.float64)

    # print(len(all_female_novel_text))
    # print(all_female_novel_text[:100])
    # print(all_female_novel_name)
    print(len(all_female_novel_text))
    print(len(unique_words))


if __name__ == "__main__":
    get_cooccurence_matrix()