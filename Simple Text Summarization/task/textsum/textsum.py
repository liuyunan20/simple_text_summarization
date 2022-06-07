import statistics
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import stopwords
import string
nltk.download('stopwords')


def sent_imp(txt):
    lemmatizer = WordNetLemmatizer()
    lemma_freq = {}
    lemma_prob = {}
    sentences_total = []
    position = 0
    lemmas_total = []
    result = []
    # lemmatize all words in text and calculate lemmas frequency
    sents = sent_tokenize(txt)
    sentence_num = round(math.sqrt(len(sents)))
    for sentence in sents:
        for word in word_tokenize(sentence.lower()):
            lemma = lemmatizer.lemmatize(word)
            if lemma not in stopwords.words('english') and lemma not in list(string.punctuation):
                lemmas_total.append(lemma)
                lemma_freq.setdefault(lemma, 0)
                lemma_freq[lemma] += 1
    # print(lemmas_total)

    # process sentences
    for sentence in sents:
        lemmas_in_sent = []
        prob_in_sent = []
        for word in word_tokenize(sentence.lower()):
            lemma = lemmatizer.lemmatize(word)
            if lemma not in stopwords.words('english') and lemma not in list(string.punctuation):
                lemmas_in_sent.append(lemma)
                lemma_prob.setdefault(lemma, lemma_freq[lemma]/len(lemmas_total))
                prob_in_sent.append(lemma_prob[lemma])
        sentences_total.append([position, statistics.mean(prob_in_sent), lemmas_in_sent, sentence])
        position += 1

    while len(result) < sentence_num:
        sentences_total.sort(key=lambda x: x[1], reverse=True)
        highest_lemma = sorted(list(lemma_prob.items()), key=lambda x: x[1], reverse=True)[0][0]
        for sentence in sentences_total:
            if highest_lemma in sentence[2] and sentence not in result:  # sentence[2] stores a list of lemmas in sentence
                result.append(sentence)
                for lemma in sentence[2]:
                    lemma_prob[lemma] = lemma_prob[lemma] ** 2
                break
        for sentence in sentences_total:
            sentence[1] = statistics.mean([lemma_prob[x] for x in sentence[2]])

    return [x[3] for x in sorted(list(result), key=lambda x: x[0])]


file = open('news.xml', 'r').read()
soup = BeautifulSoup(file, 'xml')
headers = soup.find_all('value', {'name': 'head'})
texts = soup.find_all('value', {'name': 'text'})
for header, text in zip(headers, texts):
    print(f'HEADER: {header.text}')
    sentences = sent_imp(text.text)
    print('TEXT: {}'.format("\n".join(sentences)))

