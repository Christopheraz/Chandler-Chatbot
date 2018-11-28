import sys
import os
import re
import chardet

def main():
    lastQ = ""
    ChanA = ""
    for line in sys.stdin:
        try:
            line = line.decode('ascii').strip('\n')
        except:
            continue
        #print chardet.detect(line)
        if line.startswith('Chan') == False and line.startswith('CHAN') == False and line.startswith('All') == False and line.startswith('ALL') == False:
            if line.startswith('(') == False and line.startswith('[') == False:
                listQ = line.split(' ')
                lastQ = ' '.join(listQ[1:])
        else:
            listA = line.split(' ')
            ChanA = ' '.join(listA[1:])
            if len(lastQ) > 200:
                lastQ = lastQ[:200]
            if len(ChanA) > 200:
                ChanA = ChanA[:200]
            try:
                print lastQ.encode('utf-8') + '\t' + ChanA.encode('utf-8')
            except:
                continue



if __name__ == '__main__':
    main()
