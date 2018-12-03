import sys

for line in sys.stdin:
    line = line.strip()
    em_list = line.split('\t')
    print em_list[0] + '\t' + em_list[3]
