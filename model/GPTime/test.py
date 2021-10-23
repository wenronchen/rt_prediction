import argparse
import numpy as np
import pickle
import GPTime
import csv

def parse_commandline():
    parser = argparse.ArgumentParser(description="GPTime Train")
    parser.add_argument('-p','--peptides', help="Path to the file containing their peptides and retention time", required=True)
    parser.add_argument('-m','--model', help="""Path to the model file."""
                                                , required=True)
    args = parser.parse_args()

    return args

if __name__=="__main__" :
    args = parse_commandline()

    peptides = GPTime.peptides.load(args.peptides, check_duplicates=False)
    model = GPTime.model.load( args.model )
	
    with open(args.model[args.model.find('model')+6:args.model.find('.pkl')]+'_output.tsv','wt') as out_file:
        tsv_writer=csv.writer(out_file,delimiter='\t')
        tsv_writer.writerow(['sequence','actual_rt','predicted_rt','predicted_variance','predicted_std'])
        for p in peptides :
            m,v = model.eval(p)
            s = np.sqrt(v)
            
            tsv_writer.writerow([p.sequence, p.rt,m,v,s])
        	
        
        
        #print(p.sequence, p.rt,m,v,s)
