import re       
names = input()
print(re.split(r'[0-9]+', names))
