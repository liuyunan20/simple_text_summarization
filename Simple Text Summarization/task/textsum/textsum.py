import statistics
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
nltk.download('stopwords')


def preprocess(head):
    lemmatizer = WordNetLemmatizer()
    lemmas_in_head = []
    for word in word_tokenize(head):
        word = word.lower()
        if word not in stopwords.words('english') and word not in list(string.punctuation):
            lemma = lemmatizer.lemmatize(word)
            lemmas_in_head.append(lemma)
    return lemmas_in_head


def sent_imp(txt, extra):
    lemmatizer = WordNetLemmatizer()
    model = TfidfVectorizer(tokenizer=word_tokenize)
    sentences_total = []
    position = 0
    sents_in_news = []
    # lemmatize words and build database
    sents = sent_tokenize(txt)
    sentence_num = round(math.sqrt(len(sents)))
    for sentence in sents:
        lemmas_in_sent = []
        for word in word_tokenize(sentence):
            word = word.lower()
            if word not in stopwords.words('english') and word not in list(string.punctuation):
                lemma = lemmatizer.lemmatize(word)
                lemmas_in_sent.append(lemma)
        sents_in_news.append(' '.join(lemmas_in_sent))
        sentences_total.append([position, 0, ' '.join(lemmas_in_sent), sentence])
        position += 1

    model.fit(sents_in_news)
    lemma_position = dict(model.vocabulary_)
    # print(lemma_position)
    # print(extra)
    # tfidf_matrix = model.fit_transform(sents_in_news)
    # print(tfidf_matrix)
    # process sentences
    for sentence in sentences_total:
        vector = model.transform([sentence[2]])
        term_array = vector.toarray()[0]
        for lemma in extra:
            if lemma in sentence[2] and lemma_position.get(lemma):
                term_array[lemma_position[lemma]] = term_array[lemma_position[lemma]] * 3
        term_weight = [x for x in term_array if x > 0]
        # print(term_weight)
        sentence[1] = np.mean(term_weight)

    sentences_total.sort(key=lambda x: x[1], reverse=True)
    result = sentences_total[:sentence_num]
    return [x[3] for x in sorted(list(result), key=lambda x: x[0])]


file = open('news.xml', 'r').read()
soup = BeautifulSoup(file, 'xml')
headers = soup.find_all('value', {'name': 'head'})
texts = soup.find_all('value', {'name': 'text'})
for header, text in zip(headers, texts):
    print(f'HEADER: {header.text}')
    words_in_header = preprocess(header.text)
    sentences = sent_imp(text.text, words_in_header)
    print('TEXT: {}'.format("\n".join(sentences)))

