import sys

def main():
    max_score = 0.0
    sum_score = 0.0
    num = 0
    last_Q = ''
    f = open('angry_score')
    angre_dict = {}
    for line in f:
        line = line.strip('\n ')
        ang_list = line.split('\t')
        angre_dict[ang_list[0]] = ang_list[1]
    for line in sys.stdin:
        if line.startswith('Q'):
            last_Q = line.strip('\n').strip('<92>')
        if line.startswith('A'):
            if len(line) > 100:
                continue
            ang_score = 0.0
            line = line[3:].strip('\n').strip('<92>')
            hehe = line.strip('\n ').split(' ')
            for w in hehe:
                if w in angre_dict:
                    ang_score = ang_score + float(angre_dict[w])
            if ang_score > 0.55:
                print last_Q
                print 'A::' + line
            #if ang_score > max_score:
                #max_score = ang_score
            #sum_score = sum_score + ang_score
            #num = num + 1
    #print max_score
    #print num
    #print sum_score/num



if __name__ == '__main__':
    main()
