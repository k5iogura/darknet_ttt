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

k='train'
if k in image.keys() and image[k].shape[1]==2:
    print("label file:check %s key for bitwise posi/nega"%k)
    print(image[k].shape)
    truth = np.transpose(image[k],(1,0,-1))[0]
    posi = len(truth[truth==1.])
    nega = len(truth[truth==0.])
    print("posi/nega = %.1f%% %d/%d"%(float(100.*posi)/(posi+nega),posi,nega))
