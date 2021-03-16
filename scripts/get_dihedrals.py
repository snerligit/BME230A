import re
import os
import sys
import argparse
from pymol import cmd

def parse_args():

    parser = argparse.ArgumentParser(description='Program to get dihedrals of the peptide for Ramachandran plots')
    parser.add_argument("-dir", help="directory containing Trimmed PDB templates")
    parser.add_argument("-rmsd", help="read rmsd file to get the PDB IDs")
    parser.add_argument("-pep_len", help="peptide length of interest", default=9, type=int)
    return parser.parse_args()

def read_rmsd(args):

    pep_pdbs = {}
    inputfilehandler = open(args.rmsd, 'r')
    header = inputfilehandler.readline()
    for line in inputfilehandler:
        line = line.rstrip()
        fields = line.split(',')
        ids = fields[0].split('_')
        pdb = ids[0]
        pep = ids[1]
        if len(pep) == args.pep_len:
            pep_pdbs[pdb] = pep
    inputfilehandler.close()
    return pep_pdbs

def main():

    args = parse_args()
    pep_pdbs = read_rmsd(args)

    for i in range(0, 7):
        outputfilehandler = open('phi_psi_'+str(i+2)+'.csv', 'w')
        outputfilehandler.write("pdb,chain,pep_seq,index,phi,psi\n")
        outputfilehandler.close()

    for key in pep_pdbs:
        phi = []
        psi = []
        model_name = key+'_reordered'
        pdbfile = key+'_reordered.pdb'
        targetfile = args.dir+'/'+pdbfile

        cmd.load(targetfile)
        chains = cmd.get_chains(model_name)

        try:
            #psi.append(cmd.get_dihedral(chains[1]+"/1/N,",chains[1]+"/1/ca,",chains[1]+"/1/c,",chains[1]+"/2/n"))
            #phi_psi = cmd.phi_psi(model_name+ " and chain "+chains[1])
            for i in range(2,9):
                phi.append(cmd.get_dihedral(chains[1]+"/"+str(i-1)+"/c,",chains[1]+"/"+str(i)+"/n,",chains[1]+"/"+str(i)+"/ca,",chains[1]+"/"+str(i)+"/c"))
                psi.append(cmd.get_dihedral(chains[1]+"/"+str(i)+"/N,",chains[1]+"/"+str(i)+"/ca,",chains[1]+"/"+str(i)+"/c,",chains[1]+"/"+str(i+1)+"/n"))

            #phi.append(cmd.get_dihedral(chains[1]+"/8/c,",chains[1]+"/9/n,",chains[1]+"/9/ca,",chains[1]+"/9/c"))

        except:
            print ("Found exception, ignoring")

        if len(phi) == 7 and len(psi) == 7:
            for i in range(0, len(phi)):
                outputfilehandler = open('phi_psi_'+str(i+2)+'.csv', 'a')
                print (pep_pdbs[key]+','+chains[1]+','+str(i+2)+','+str(phi[i])+','+str(psi[i]))
                outputfilehandler.write(key+','+chains[1]+','+pep_pdbs[key]+','+str(i+2)+','+str(phi[i])+','+str(psi[i])+'\n')
                outputfilehandler.close()


        pymol.cmd.do("delete all")


if __name__ == "__main__":

    main()
