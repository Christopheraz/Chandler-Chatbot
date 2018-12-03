import sys
import os
import re
import chardet
import nltk
import nltk.data

ChanA = ''
toChan = ''
lastQ = ''
toC = ''

def process_line(line):
    line = re.sub('\([\w\W]+?\)', '', line)
    str_list = line.split(':')
    new_line = ''.join(str_list[1:])
    new_list = splitSentence(new_line)
    if len(new_list) > 4:
        new_list = new_list[:4]
    final_line = ' '.join(new_list)
    if final_line[0] == ' ':
        final_line = final_line[1:]
    return final_line


def printQA(toc, lastq, chana):
    try:
        chA = process_line(chana)
        if toc != '':
            print 'Q::' + toc.encode('utf-8')
            print 'A::' + chA.encode('utf-8')
        if lastq != '':
            laQ = process_line(lastq)
            print 'Q::' + laQ.encode('utf-8')
            print 'A::' + chA.encode('utf-8')
    except:
        return


def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def main():
    for line in sys.stdin:
        try:
            line = line.decode('ascii').strip('\n')
        except:
            continue
        line = line.lower()
        if line.startswith('[') or line.startswith('end'):
            lastQ = ''
            ChanA = ''
            toChan = ''
            continue
        if line.startswith('('):
            continue
        if ':' in line:
            if 'to chandler' in line:
                toChan = line
                toC_list = toChan.split('to chandler')
                toChan = ' '.join(toC_list[1:])
                temp = 0
                for i in toChan:
                    if i == ')' or i == ']':
                        temp = 1
                        continue
                    if i == '(' or i == '[':
                        temp = 0
                        break
                    if temp == 1:
                        toC = toC + i
                        continue
                try:
                    if toC[0] == ' ':
                        toC = toC[1:]
                except:
                    continue
                toChan = toC
                toC = ''
                continue
            if line.startswith('chan'):
                if lastQ == '' and toChan == '':
                    continue
                ChanA = line
                #print ChanA
                printQA(toChan, lastQ, ChanA)
                lastQ = ''
                toChan = ''
                ChanA = ''
                toC = ''
            else:
                lastQ = line



if __name__ == '__main__':
    main()
