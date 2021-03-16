import os
import re
import sys
import pymol
import argparse

from fasta import FASTA

'''
python pairwise_rmsd_matrix_pdb_peptides.py -dir TRIMMED_TEMPLATES/ -fasta PDBS.fasta
'''

def get_args():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description="Method to process ia score files")
    parser.add_argument("-dir", help="directory containing the PDB template files")
    parser.add_argument("-fasta", help="PDBS.fasta")

    args = parser.parse_args()

    return args

def read_fasta(args):

    fasta = FASTA(args.fasta)
    fasta.read()
    headers = fasta.get_headers()
    pep_chain = {}
    pep_seq = {}

    for header in headers:
        fields = header.split('|')
        pdbid = fields[0]
        chainid = fields[1]
        seq = fasta.get_sequence(header)

        if len(seq) == 9:
            pep_chain[pdbid] = chainid
            pep_seq[pdbid] = seq

    return (pep_chain, pep_seq)

def get_rmsd(dir, template_pdb, template_chain, target_pdb, target_chain):
    pymol.cmd.load(dir+'/'+template_pdb+'.pdb', "obj01")
    pymol.cmd.load(dir+'/'+target_pdb+'.pdb', "obj02")

    chain1 = "obj01 and chain "+template_chain+" and name n+c+ca+o"
    chain2 = "obj02 and chain "+target_chain+" and name n+c+ca+o"
    if template_chain != target_chain:
        pymol.cmd.do("alter "+chain2+", chain=\'"+template_chain+"\'")
        chain3 = "obj02 and chain "+template_chain+" and name n+c+ca+o"
        #print (chain1, chain3)
        pymol.cmd.super("obj01", "obj02", cycles=0)
        rms = pymol.cmd.rms_cur(chain1, chain3, cycles=0, matchmaker=4)
    else:
        pymol.cmd.super("obj01", "obj02", cycles=0)
        rms = pymol.cmd.rms_cur(chain1, chain2, cycles=0, matchmaker=4)
    pymol.cmd.do("delete all")

    return rms

'''
def get_label(rmsd):

    if rmsd <= 1.0:
        return 0
    elif rmsd > 1.0 and rmsd <= 1.5:
        return 1
    elif rmsd > 1.5 and rmsd <= 2.0:
        return 2
    else:
        return 3

'''

def get_label(rmsd):

    if rmsd <= 1.5:
        return 0
    else:
        return 1

def process(args, pep_chain, pep_seq):

    rmsoutputfilehandler = open("rms_output.csv", 'w')
    rmsoutputfilehandler.write("t1,t2,rms,label\n")

    pdbs = pep_chain.keys()

    rms_str = []
    for pdb1 in pdbs:
        template_file_name = pdb1+"_reordered.pdb"
        template_file = args.dir+'/'+template_file_name
        for pdb2 in pdbs:
            target_file_name = pdb2+"_reordered.pdb"
            target_file = args.dir+'/'+target_file_name
            template_pdb = template_file_name.split('.pdb')[0]
            target_pdb = target_file_name.split('.pdb')[0]
            template_pdbid = template_file_name.split('_')[0]
            target_pdbid = target_file_name.split('_')[0]
            rms = get_rmsd(args.dir, template_pdb, pep_chain[template_pdbid], target_pdb, pep_chain[target_pdbid])
            my_string = template_pdbid+'_'+pep_seq[template_pdbid]+','+target_pdbid+'_'+pep_seq[target_pdbid]+','+str(rms)+','+str(get_label(rms))
            rms_str.append(my_string)
            rmsoutputfilehandler.write(my_string+'\n')

    rmsoutputfilehandler.close()

    for r in rms_str:
        print (r)

def main(args):

    (pep_chain, pep_seq) = read_fasta(args)
    process(args, pep_chain, pep_seq)

if __name__ == "__main__":

    main(get_args())
