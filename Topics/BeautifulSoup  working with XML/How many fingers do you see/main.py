from bs4 import BeautifulSoup

fingers = '<?xmlversion="1.0"encoding="UTF-8"?><fingers><finger><name>Thumb</name><hand>Left</hand></finger><finger><name>Index</name><hand>Left</hand></finger><finger><name>Index</name><hand>Right</hand></finger><finger><name>Middle</name><hand>Right</hand></finger><finger><name>Little</name><hand>Left</hand></finger></fingers>'

# put your code here
soup = BeautifulSoup(fingers, 'xml')
tag = soup.find_all('finger')
print(len(tag))
