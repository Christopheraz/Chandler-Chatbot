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
            newline = re.sub(r'&\w{4};', "",newline)
            newline = re.sub('</?[\w\W]{0,30}?>', '', newline)
            print newline
            newline = ''

if __name__ == '__main__':
    main()
