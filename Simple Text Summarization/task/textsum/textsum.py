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


def sent_imp(txt):
    lemmatizer = WordNetLemmatizer()
    model = TfidfVectorizer(tokenizer=word_tokenize)
    lemma_freq = {}
    lemma_prob = {}
    sentences_total = []
    position = 0
    lemmas_total = []
    result = []
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
    # print(len(sents_in_news))
    model.fit(sents_in_news)
    # tfidf_matrix = model.fit_transform(sents_in_news)
    # print(tfidf_matrix)
    # process sentences
    for sentence in sentences_total:
        vector = model.transform([sentence[2]])
        # print(vector.shape)
        term_weight = [x for x in vector.toarray()[0] if x > 0]
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
    sentences = sent_imp(text.text)
    print('TEXT: {}'.format("\n".join(sentences)))

