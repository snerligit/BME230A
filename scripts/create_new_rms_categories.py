import os
import sys
import argparse

def get_args():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description="Method to reclassify rmsds")
    parser.add_argument("-csv", help="rms_output.csv")

    args = parser.parse_args()

    return args

def relabel(args):

    inputfilehandler = open(args.csv, 'r')
    print (inputfilehandler.readline())
    for line in inputfilehandler:
        line = line.rstrip()
        fields = line.split(',')
        label = '0'
        if float(fields[2]) >= 1.5:
            label = '1'
        print (','.join(fields[0:len(fields)-1])+','+label)
    inputfilehandler.close()

def main(args):

    relabel(args)

if __name__ == "__main__":

    main(get_args())
