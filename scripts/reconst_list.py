import argparse
import os
import sys
import re
import shutil
import fnmatch
import numpy as np

parser = argparse.ArgumentParser(description='rename and copy image files for darknet classifier rule')
parser.add_argument('in_dir', nargs='+',  type=str, default='lfw/')
parser.add_argument('--posi',       '-p', type=str, default='(person|automobile)',help='posi class file rex (person|automobile) etc.')
parser.add_argument('--train_list', '-t', type=str, default='train.list')
parser.add_argument('--test_list' , '-T', type=str, default='test.list')
parser.add_argument('--verbose',    '-v', action='store_true')
args = parser.parse_args()

in_dir = args.in_dir
for in_d in in_dir:
    if not os.path.exists(in_d):
        print('%s directory not found'%in_d)
        sys.exit(1)

train_list=args.train_list
test_list =args.test_list
if os.path.exists(train_list):
    print('%s already exists'%train_list)
    sys.exit(1)
if os.path.exists(test_list):
    print('%s already exists'%test_list)
    sys.exit(1)

print('*start')
rex = args.posi
posi_files=[]
nega_files=[]
for in_d in list(in_dir):
    print('processing %s'%in_d)
    for root,dirs,names in os.walk(in_d):
        found_files = [os.path.join(root,f) for f in names
            if len(re.findall(rex,os.path.join(root,f)))>0 ]
        posi_files.extend(found_files)
        found_files = [os.path.join(root,f) for f in names
            if len(re.findall(rex,os.path.join(root,f)))==0 ]
        nega_files.extend(found_files)
print('input directories\tposi/nega=%d/%d'%(len(posi_files),len(nega_files)))

# separate 7:3
posi_filesN = len(posi_files)
nega_filesN = int(posi_filesN*3/7)
print('\t7:3 ruling\tposi/nega=%d/%d'%(posi_filesN,nega_filesN))

train_posi_filesN = int(9*posi_filesN/10)
test_posi_filesN  = posi_filesN - train_posi_filesN
train_nega_filesN = int(9*nega_filesN/10)
test_nega_filesN  = nega_filesN - train_nega_filesN
print('\t10p ruling\ttrain posi/nega=%d/%d\t test posi/nega=%d/%d'%(train_posi_filesN,train_nega_filesN,test_posi_filesN,test_nega_filesN))

print('*randomize')
np.random.seed(2222)
posi_files = np.asarray(posi_files,dtype=np.dtype('U256'))
nega_files = np.asarray(nega_files,dtype=np.dtype('U256'))
cwd=os.getcwd()
for i,f in enumerate(posi_files): posi_files[i]=cwd+'/'+posi_files[i]
for i,f in enumerate(nega_files): nega_files[i]=cwd+'/'+nega_files[i]
idx1 = np.random.permutation(posi_filesN)
idx2 = np.random.permutation(nega_filesN)
posi_files = posi_files[idx1]
nega_files = nega_files[idx2]

print('*making list')
print('train posi/nega =%d/%d'%(train_posi_filesN,train_nega_filesN))
print('test  posi/nega =%d/%d'%(test_posi_filesN,test_nega_filesN))
train_files = np.hstack((
                        posi_files[:train_posi_filesN],
                        nega_files[:train_nega_filesN]
                        ))
test_files  = np.hstack((
                        posi_files[train_posi_filesN:train_posi_filesN+test_posi_filesN],
                        nega_files[train_nega_filesN:train_nega_filesN+test_nega_filesN]
                        ))
print('train(posi+nega)/test(posi+nega) = %d/%d files'%(train_files.shape[0],test_files.shape[0]))

with open(train_list,'w') as f:
    for i,t in enumerate(train_files):
        f.write(str(t)+'\n')
    print('write out into %s/\t%d\tlines'%(train_list,i+1))

with open(test_list,'w') as f:
    for i,t in enumerate(test_files):
        f.write(str(t)+'\n')
    print('write out into %s/\t%d\tlines'%(test_list,i+1))

