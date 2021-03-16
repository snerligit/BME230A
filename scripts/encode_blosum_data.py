import argparse
from substitution_matrix import BLOSUM
from sklearn.preprocessing import OneHotEncoder

encoding = {}
encoding['11'] = ['1','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
encoding['10'] = ['0','1','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
encoding['9'] = ['0','0','1','0','0','0','0','0','0','0','0','0','0','0','0','0']
encoding['8'] = ['0','0','0','1','0','0','0','0','0','0','0','0','0','0','0','0']
encoding['7'] = ['0','0','0','0','1','0','0','0','0','0','0','0','0','0','0','0']
encoding['6'] = ['0','0','0','0','0','1','0','0','0','0','0','0','0','0','0','0']
encoding['5'] = ['0','0','0','0','0','0','1','0','0','0','0','0','0','0','0','0']
encoding['4'] = ['0','0','0','0','0','0','0','1','0','0','0','0','0','0','0','0']
encoding['3'] = ['0','0','0','0','0','0','0','0','1','0','0','0','0','0','0','0']
encoding['2'] = ['0','0','0','0','0','0','0','0','0','1','0','0','0','0','0','0']
encoding['1'] = ['0','0','0','0','0','0','0','0','0','0','1','0','0','0','0','0']
encoding['0'] = ['0','0','0','0','0','0','0','0','0','0','0','1','0','0','0','0']
encoding['-1'] = ['0','0','0','0','0','0','0','0','0','0','0','0','1','0','0','0']
encoding['-2'] = ['0','0','0','0','0','0','0','0','0','0','0','0','0','1','0','0']
encoding['-3'] = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','1','0']
encoding['-4'] = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','1']

def score_BLOSUM(template, target):
    BLOSUM62 = BLOSUM().get_matrix()
    score = []
    for i in range(0, len(template)):
        score.append(str(BLOSUM62[template[i]][target[i]]))
    return score

def find_max_min_BLOSUM():
    BLOSUM62 = BLOSUM().get_matrix()
    maxVal = -10000
    minVal = 10000
    for k1 in BLOSUM62:
        for k2 in BLOSUM62[k1]:
            val = int(BLOSUM62[k1][k2])
            if val > maxVal:
                maxVal = val
            if val < minVal:
                minVal = val

    #print ("Max: ", maxVal, "Min: ", minVal)

def oneHotEncoding(score):

    codes = []
    for s in score:
        codes += encoding[s]

    return codes


def read(args):

    inputfilehandler = open(args.infile, 'r')
    find_max_min_BLOSUM()
    header = inputfilehandler.readline()
    for line in inputfilehandler:
        line = line.rstrip()
        fields = line.split(',')
        temp = fields[0].split('_')[1]
        target = fields[1].split('_')[1]
        rms = float(fields[2])
        score = score_BLOSUM(temp, target)
        label = '0'
        #label = 'yes'
        if rms > 1.5:
            label = '1'
            #label = 'no'
        codes = oneHotEncoding(score)
        print (len(codes))
        #print (', '.join(codes))
        #print (label)

    inputfilehandler.close()

def get_args():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description="Method to process ia score files")
    parser.add_argument("-infile", help="input file containing template and target peptide sequences")


    args = parser.parse_args()

    return args

def main(args):

    read(args)


if __name__ == "__main__":

    main(get_args())
