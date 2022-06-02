from gender_novels.corpus import Corpus
def get_corpus():
    print("Loading corpus")
    corpus = Corpus('gutenberg')
    female_corpus = corpus.filter_by_gender('female')
    male_corpus = corpus.filter_by_gender('male')
    return female_corpus, male_corpus
def get_cooccurence_matrix():
    female_corpus, male_corpus = get_corpus()
    all_female_novel_text = []
    all_female_novel_name = []
    for novel in female_corpus.novels:
        all_female_novel_name.append(novel.filename)
        all_female_novel_text.extend(novel.get_tokenized_text())
    
    print(len(all_female_novel_text))
    print(all_female_novel_text[:100])
    print(all_female_novel_name)
if __name__ == "__main__":
    get_cooccurence_matrix()

















