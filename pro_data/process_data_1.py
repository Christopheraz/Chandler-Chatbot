import sys
import os
import re
import nltk

def main():
    temp = 0
    newline = ""
    num = 0
    for line in sys.stdin:
        line = line.strip('\n')
        newstring = ""
        for i in line:
            if i == '<':
                temp = 1
            elif i == '>':
                temp = 0
                continue
            if temp == 0:
                newstring = newstring + i

        if newstring == "":
            newline = re.sub(r'&[\w\W]{1,10};',"",newline)
            newline = newline.replace('&#8217;','\'')
            newline = re.sub(r'&#[\w\W]{1,10};',"",newline)
            if newline != "":
                num = num + 1
                if num > 3:
                    print newline[1:]
            newline = ""
        else:
            newline = newline + " " + newstring.strip('\n')




if __name__ == '__main__':
    main()
