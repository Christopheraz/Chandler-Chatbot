import sys
import os
import re

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
            newline = re.sub(r'&\w{4};',"",newline)
            if newline != "":
                num = num + 1
                if num > 4:
                    print newline.strip()
            newline = ""
        else:
            newline = newline + " " + newstring.strip('\n')




if __name__ == '__main__':
    main()
