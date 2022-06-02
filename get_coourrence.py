import re
from nltk.tokenize import sent_tokenize, word_tokenize
from gender_novels.corpus import Corpus
def get_corpus():
    print("Loading corpus")
    corpus = Corpus('sample_novels')
    female_corpus = corpus.filter_by_gender('female')
    male_corpus = corpus.filter_by_gender('male')
    return female_corpus, male_corpus

def get_cooccurence_matrix():
    female_corpus, male_corpus = get_corpus()
    all_female_novel_text = []
    all_female_novel_name = []
    sentence_words_list = []
    pattern = re.compile(r"\w+(?:-\w+)+")

    for novel in female_corpus.novels:
        sentences = sent_tokenize(novel.text.lower().replace("\n", " "))
        #  print(sentences[:10])
        for sentence in sentences:
            words = [word for word in word_tokenize(sentence) if pattern.match(word) or word.isalpha()]
            sentence_words_list.append(words)
        #  print(sentence_words_list[:10])
        all_female_novel_name.append(novel.filename)
        all_female_novel_text.extend(novel.get_tokenized_text())
    
    print(len(all_female_novel_text))
    print(all_female_novel_text[:100])
    print(all_female_novel_name)
if __name__ == "__main__":
    get_cooccurence_matrix()

















