import sys
import os
import re
import json

def main():
    with open('CAI2.json', 'r') as f:
        data = json.load(f)
    for i in data:
        for j in i[u'dialog']:
            print j
    #print data
if __name__ == '__main__':
    main()
