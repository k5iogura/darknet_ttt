import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='check Original dataset and annotation into pickle')
parser.add_argument('--ds_file', '-d', type=str, default='voc_ds.pkl')
args = parser.parse_args()

with open(args.ds_file,'rb') as f:
    print("analized:%s"%args.ds_file)
    image=pickle.load(f)

print("Usable image ndarray")
print(image.keys())

for k in image.keys():
    print("key=%s\t%d"%(k,len(image[k])))
