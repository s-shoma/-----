import requests
import re

r = requests.get('https://www.python.org/downloads/')

release = []
for li in re.findall(r'<li>.+?</li>',
                     r.text.replace('¥n', '')):
    if x := re.search(r'<span class="release-number">'
                      r'<a href=".+?">(.+?)</a></span>', li):
        if y := re.search(r'<span class="release-date">'
                          r'(.+?)</span>', li):
            release.append((x.group(1), y.group(1)))

release.sort()
for name, date in release:
    print(f'{name:15}{date}')