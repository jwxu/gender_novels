import numpy as np
from gender_novels.corpus import Corpus
from collections import defaultdict


def get_corpus():
    print("Loading corpus")
    corpus = Corpus('sample_novels')
    female_corpus = corpus.filter_by_gender('female')
    male_corpus = corpus.filter_by_gender('male')

    return female_corpus, male_corpus


def get_top_words(corpus, top_n_words=10000):
    all_novel_text = []
    for novel in corpus.novels:
        all_novel_text.extend(novel.get_tokenized_text())

    word_counts = defaultdict(int)
    for word in all_novel_text:
        word_counts[word] += 1

    top_words_count = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n_words]
    top_words = [word_count[0] for word_count in top_words_count]

    return top_words


def get_cooccurence_matrix(corpus, target_words):
    # Get novel text (as list of words) and file title
    all_novel_text = []
    all_novel_name = []
    for novel in corpus.novels:
        all_novel_name.append(novel.filename)
        all_novel_text.extend(novel.get_tokenized_text())
    
    num_unique = len(target_words)

    cooc_mat = np.zeros((num_unique, num_unique),np.float64)


if __name__ == "__main__":
    female_corpus, male_corpus = get_corpus()
    female_cooccur_words = get_top_words(female_corpus, top_n_words=10000)
    male_cooccur_words = get_top_words(male_corpus, top_n_words=10000)

    # get_cooccurence_matrix(female_corpus, female_cooccur_words)