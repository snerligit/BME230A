import os
import sys
import numpy
import argparse

from fasta import FASTA

from pyaaisc import Aaindex

labelhydrophobiccodes = {}
labelhydrophobiccodes['0'] = ['0','0','0','0']
labelhydrophobiccodes['1'] = ['0','1','0','0']
labelhydrophobiccodes['2'] = ['0','0','1','0']
labelhydrophobiccodes['3'] = ['0','0','0','1']

#labelhydrophobiccodes['0'] = ['0']
#labelhydrophobiccodes['1'] = ['1']


def parse_args():

    parser = argparse.ArgumentParser(usage="Please see the options below")
    parser.add_argument("-dir", help="Directory containing trimmed PDB file, TRIMMED_TEMPLATES")
    parser.add_argument("-fasta", help="PDBS.fasta file")
    parser.add_argument("-grooves", help="grooves.txt file")
    parser.add_argument("-rms", help="rms_output.csv file")
    parser.add_argument("-t", help="train or test file containing PDB names")
    parser.add_argument("-label", help="print x or y labels")
    parser.add_argument("-pep", help="input feature is a pep only", default=False)

    return parser.parse_args()

def totalNineMers(fasta):

    headers = fasta.get_headers()
    nineMers = 0
    peptides = {}
    mhcSeq = {}
    mhcAllele = {}
    for header in headers:
        seq = fasta.get_sequence(header)
        fields = header.split('|')
        pdbid = fields[0]
        if len(seq) == 9:
            nineMers += 1
            allele = fields[2]
            peptides[pdbid] = seq
            mhcAllele[pdbid] = allele
        elif len(seq) >= 179:
            mhcSeq[pdbid] = seq

    #print ("Total 9-mers in PDB: ", nineMers)
    return (peptides, mhcSeq, mhcAllele)

def universalGroove(groove, mhcSeq, peptides):

    inputfilehandler = open(groove, 'r')
    universalGroove = []
    grooves = {}
    for line in inputfilehandler:
        line = line.rstrip()
        fields = line.split()
        pdbid = fields[0]
        if pdbid in peptides:
            resids = fields[1].split(',')
            for r in resids:
                if r not in universalGroove:
                    universalGroove.append(r)

    inputfilehandler.close()

    for pdbid in peptides:
        seq = mhcSeq[pdbid]
        grooveSeq = []
        for r in universalGroove:
            grooveSeq.append(seq[int(r)])
        grooves[pdbid] = ''.join(grooveSeq)

    return grooves

def IntersectionGroove(groove, mhcSeq, peptides):

    inputfilehandler = open(groove, 'r')
    intersectGroove = ['5','7','9','33','45','59','62','62','63','66','67','69','70','73','74','76','77','80','81','84','95','97','99','116','123','124','143','146','147','152','155','156','159','167','171']
    grooves = {}
    for line in inputfilehandler:
        line = line.rstrip()
        fields = line.split()
        pdbid = fields[0]
        if pdbid in peptides:
            resids = fields[1].split(',')
            for r in intersectGroove:
                if r not in resids:
                    intersectGroove.remove(r)

    inputfilehandler.close()

    for pdbid in peptides:
        seq = mhcSeq[pdbid]
        grooveSeq = []
        for r in intersectGroove:
            grooveSeq.append(seq[int(r)])
        grooves[pdbid] = ''.join(grooveSeq)

    return grooves

def readGrooves(groove, mhcSeq, peptides):

    inputfilehandler = open(groove, 'r')
    grooves = {}
    for line in inputfilehandler:
        line = line.rstrip()
        fields = line.split()
        pdbid = fields[0]
        if pdbid in peptides:
            resids = fields[1].split(',')
            seq = mhcSeq[pdbid]
            grooveSeq = []
            for r in resids:
                grooveSeq.append(seq[int(r)])
            grooves[pdbid] = ''.join(grooveSeq)
    inputfilehandler.close()
    return grooves

def read_rmsd_file(rms):

    labels = {}
    inputfilehandler = open(rms, 'r')
    header = inputfilehandler.readline()
    for line in inputfilehandler:
        line = line.rstrip()
        fields = line.split(',')
        model1 = fields[0]
        model2 = fields[1]
        pdbid1 = model1.split('_')[0]
        pdbid2 = model2.split('_')[0]
        rmsd = fields[2]
        classLabel = fields[3]
        labels[pdbid1+'_'+pdbid2] = classLabel
    inputfilehandler.close()
    return labels

def read_datafile(datafile):

    pdbids = []
    inputfilehandler = open(datafile, 'r')
    for line in inputfilehandler:
        line = line.rstrip()
        pdbids.append(line)
    inputfilehandler.close()
    return pdbids

def gethydrophobiccode(aa, index_data):

    value = index_data[aa]
    code = []
    if value >= -7 and value < -6:
        code = ['1','0','0','0','0','0','0','0','0']
    elif value >= -5 and value < -4:
        code = ['0','1','0','0','0','0','0','0','0']
    elif value >= -4 and value < -3:
        code = ['0','0','1','0','0','0','0','0','0']
    elif value >= -2 and value < -1:
        code = ['0','0','0','1','0','0','0','0','0']
    elif value >= -1 and value < 0:
        code = ['0','0','0','0','1','0','0','0','0']
    elif value >= 0 and value < 1:
        code = ['0','0','0','0','0','1','0','0','0']
    elif value >= 1 and value < 2:
        code = ['0','0','0','0','0','0','1','0','0']
    elif value >= 2 and value < 3:
        code = ['0','0','0','0','0','0','0','1','0']
    elif value >= 3 and value < 4:
        code = ['0','0','0','0','0','0','0','0','1']

    return code

def getchargecode(aa, charge_data):

    value = charge_data[aa]
    code = []
    if value == -1:
        code = ['1','0','0']
    elif value == 0:
        code = ['0','1','0']
    elif value == 1:
        code = ['0','0','1']

    return code

def oneHotEncoding(sequence, label, index_data, charge_data):

    finalSeqCode = []
    finalLabelCode = []
    for s in sequence:
        if s == '|':
            finalSeqCode += ['0']
        else:
            #finalSeqCode += gethydrophobiccode(s, index_data) + getchargecode(s, charge_data)
            finalSeqCode += gethydrophobiccode(s, index_data)

    for l in label:
        finalLabelCode += labelhydrophobiccodes[l]

    #print (len(finalSeqCode))

    return finalSeqCode, finalLabelCode


def main():

    args = parse_args()

    fasta = FASTA(args.fasta)
    fasta.read()
    (peptides, mhcSeq, mhcAllele) = totalNineMers(fasta)
    #grooves = readGrooves(args.grooves, mhcSeq, peptides)
    universalGrooves = universalGroove(args.grooves, mhcSeq, peptides)
    intersectGrooves = IntersectionGroove(args.grooves, mhcSeq, peptides)
    labels = read_rmsd_file(args.rms)
    pdbids = read_datafile(args.t)

    aaindex = Aaindex()
    #for result in aaindex.search('charge'):
    #    print(result)

    record = aaindex.get('FASG890101')
    #print (record.title)
    index_data = record.index_data
    #print (index_data)

    charge = aaindex.get('KLEP840101')
    charge_data = charge.index_data
    #print (charge_data)

    for l in labels:
        (pdbid1, pdbid2) = l.split('_')
        #if pdbid1 in pdbids and pdbid2 in pdbids:
        if pdbid1 in pdbids or pdbid2 in pdbids:
            if args.pep:
                finalSeqCode, finalLabelCode = oneHotEncoding(peptides[pdbid1]+'|'+peptides[pdbid2], labels[l], index_data, charge_data)
                if args.label == 'x':
                    print (', '.join(finalSeqCode))
                elif args.label == 'y':
                    print (', '.join(finalLabelCode))
            else:
                finalSeqCode, finalLabelCode = oneHotEncoding(universalGrooves[pdbid1]+peptides[pdbid1]+'|'+universalGrooves[pdbid2]+peptides[pdbid2], labels[l], index_data, charge_data)
                if args.label == 'x':
                    print (', '.join(finalSeqCode))
                elif args.label == 'y':
                    print (', '.join(finalLabelCode))


if __name__ == "__main__":

    main()
