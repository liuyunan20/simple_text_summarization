import re


# put your regex in the variable template
template = r"[0-3]?[0-9][/\.][01]?[0-9][/\.][0-9]{4}"
string = input()
# compare the string and the template
match = re.match(template, string)
if match:
    print(match.group()[-4:])
else:
    print(None)
