import sys

def main():
    for line in sys.stdin:
        line = line.strip('\n')
        print 'python process_data_1.py < ../data/' + line

if __name__ == '__main__':
    main()
