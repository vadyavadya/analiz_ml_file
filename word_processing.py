import unicodedata

import nltk
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.ru import Russian
from sklearn.feature_extraction.text import TfidfVectorizer



class WordProcessing(object):

    # def __init__(self):

    def tokenize_sentence(self, sentence):
        """
        Сегментирует, лексемизирует и маркирует документ в корпусе.
        """
        for paragraph in iter(sentence.splitlines()):
            if paragraph:
                yield [
                    pos_tag(wordpunct_tokenize(sent), lang='rus')
                    for sent in sent_tokenize(paragraph)
                ]

    def tokenize_text(self, text):
        text_tokenize = []
        sentence_tokenize = []
        for n in range(len(text)):
            for sentence in self.tokenize_sentence(text[n][0]):
                sentence_tokenize.extend(sentence)
            text_tokenize.append(sentence_tokenize.copy())
            sentence_tokenize.clear()
        return text_tokenize


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='russian'):
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = Russian()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag, self.nlp).lower()
            # for paragraph in document
            for sentence in document  # paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token) and tag != 'NUM=ciph'
        ]

    def lemmatize(self, token, pos_tag, nlp):
        docs = iter(nlp(token))
        return next(docs).lemma_

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        text_normaliz = []
        for document in documents:
            text_normaliz.append(self.normalize(document))  # новая версия со справочником
        return text_normaliz


class TextVectorizer(TfidfVectorizer):

    def build_vocubulaey(self, text):
        return self.fit(text)

    def text_vectorizer(self, text):
        return self.transform(text)


def identity(words):
    return words

