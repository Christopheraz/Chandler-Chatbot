import sys
import os
import re

def main():
    temp = 0
    item = 0
    newline = ""
    num = 0
    for line in sys.stdin:
        line = line.strip('\n')
        if line.startswith('<p') == True:
            newline = newline + line.strip('\n')
        if line.startswith('<') == False:
            newline = newline + line.strip('\n')
        if line.endswith('</p>') == True:
            newline = re.sub(r'&\w{1,10};', "",newline)
            newline = newline.replace('&#8217;','\'')
            newline = re.sub(r'&#[\w\W]{1,10};',"",newline)
            newline = re.sub('</?[\w\W]{0,30}?>', '', newline)
            newlist = newline.split(' ')
            for t in newlist:
                t = t.strip(' ')
            print ' '.join(newlist)
            newline = ''

if __name__ == '__main__':
    main()
