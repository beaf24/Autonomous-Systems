import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse


parent_dir = os.getcwd()
mapp = np.loadtxt(parent_dir + "/data/export.txt", delimiter=",")


parser = argparse.ArgumentParser()
parser.add_argument("-p_free", "--pfree", type=str)
args = parser.parse_args()



prob_map = 1 - 1/(1+np.exp(mapp))

# adicionado para thresholding
prob_map[prob_map>=0.75]=1
prob_map[prob_map<=0.25]=0
prob_map[(prob_map > 0.25) & (prob_map < 0.75)] = 0.5



IM = Image.fromarray(255-np.uint8(prob_map*255))
#IM=IM.rotate(90, expand=1,fillcolor= "grey")

IM.save(parent_dir + "/images/map_algo-" + args.pfree + ".png") 
            
