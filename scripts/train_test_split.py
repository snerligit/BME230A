import os
import sys
import numpy
from random import random
import argparse

from fasta import FASTA


def parse_args():

    parser = argparse.ArgumentParser(usage="Please see the options below")
    parser.add_argument("-fasta", help="PDBS.fasta file")
    parser.add_argument("-percent", help="random number for train test splitting", type=float, default=0.1)

    return parser.parse_args()

def totalNineMers(fasta):

    headers = fasta.get_headers()
    peptides = {}
    for header in headers:
        seq = fasta.get_sequence(header)
        fields = header.split('|')
        pdbid = fields[0]
        if len(seq) == 9:
            allele = fields[2]
            peptides[pdbid] = seq

    return peptides

def write_to_file(infile, dataset):

    outputfilehandler = open(infile, 'w')
    for d in dataset:
        outputfilehandler.write(d+'\n')
    outputfilehandler.close()

def main():

    args = parse_args()

    fasta = FASTA(args.fasta)
    fasta.read()
    peptides = totalNineMers(fasta)
    pdbids = peptides.keys()
    testsetlen = int(args.percent * len(pdbids))

    trainset = []
    testset = []
    for i in range(0, len(pdbids)):
        r = random()
        if len(testset) < testsetlen and r < 0.5:
            testset.append(pdbids[i])
        else:
            trainset.append(pdbids[i])

    write_to_file('train/90_10/train.txt', trainset)
    write_to_file('test/90_10/test.txt', testset)


if __name__ == "__main__":

    main()
