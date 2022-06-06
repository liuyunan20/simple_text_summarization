from bs4 import BeautifulSoup

apartment = '<?xml version="1.0" encoding="utf-8"?><apartment><living_room><furniture><item number="5">chair</item><item number="1">sofa</item><item number="2">table</item><item number="3">bookcase</item><item number="2">armchair</item></furniture><other><item number="1">fireplace</item></other></living_room><bedroom><furniture><item number="1">bed</item><item number="2">bedside_cabinet</item></furniture><other><item number="2">lamp</item></other></bedroom></apartment>'

# put your code here
soup = BeautifulSoup(apartment, 'xml')
tag = soup.find_all('living_room')
for t in tag:
    print(list(t.children))
