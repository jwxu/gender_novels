import numpy as np
from gender_novels.corpus import Corpus
from collections import defaultdict
import itertools
from scipy.sparse import csr_matrix


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

def create_co_occurences_matrix(allowed_words, documents):
    print(f"allowed_words:\n{allowed_words}")
    print(f"documents:\n{documents}")
    word_to_id = dict(zip(allowed_words, range(len(allowed_words))))
    documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in documents]
    row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
    data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
    max_word_id = max(itertools.chain(*documents_as_ids)) + 1
    docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))  # efficient arithmetic operations with CSR * CSR
    words_cooc_matrix = docs_words_matrix.T * docs_words_matrix  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
    words_cooc_matrix.setdiag(0)
    print(f"words_cooc_matrix:\n{words_cooc_matrix.todense()}")
    return words_cooc_matrix, word_to_id

if __name__ == "__main__":
    female_corpus, male_corpus = get_corpus()
    female_cooccur_words = get_top_words(female_corpus, top_n_words=10000)
    male_cooccur_words = get_top_words(male_corpus, top_n_words=10000)

    # get_cooccurence_matrix(female_corpus, female_cooccur_words)