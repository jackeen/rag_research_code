# import spacy
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")

if __name__ == "__main__":
    # nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # s = "book books Books Book BOOK BOOKS"
    # s = s.lower()
    # s = str(s)

    # docs = nlp(s)

    # for token in docs:
    #     print(token.lemma_)

    lemmatizer = WordNetLemmatizer()
    words = ["flexible layouts", "media queries"]
    for w in words:
        ww = lemmatizer.lemmatize(w, pos="n")
        print(ww)
