from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import math


file = open('news.xml', 'r').read()
soup = BeautifulSoup(file, 'xml')
headers = soup.find_all('value', {'name': 'head'})
texts = soup.find_all('value', {'name': 'text'})
for header, text in zip(headers, texts):
    print(f'HEADER: {header.text}')
    sentences = sent_tokenize(text.text)
    n = round(math.sqrt(len(sentences)))
    print('TEXT: {}'.format("\n".join(sentences[:n])))
