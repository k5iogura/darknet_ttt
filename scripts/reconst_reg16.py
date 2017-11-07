import pickle
import numpy as np
import os
import sys
import argparse
import fnmatch
from pdb import *

files = ['voc_ds.pkl', 'lfw_ds.pkl', 'indoorCVPR_09_ds.pkl']

parser = argparse.ArgumentParser(description='check Original dataset and annotation into pickle')
parser.add_argument('--ds_files',   '-d',  nargs='+',type=str)
parser.add_argument('--no_shuffle', '-ns', action='store_false')
parser.add_argument('--use_nega_all','-una',action="store_true")
parser.add_argument('--truth_center','-tc',action="store_true")
parser.add_argument('--truth_manhat','-tm',action="store_true")
parser.add_argument('--only_check',  '-oc',action="store_true")
args = parser.parse_args()

if args.ds_files:
    for ds_file in args.ds_files:
        if os.path.exists(ds_file):
            files.append(ds_file)
        else:
            print('%s'%os.path.splitext(ds_file)[1])
            print('append %s file not found'%ds_file)
            sys.exit(1)
check_file=0
for ds_file in files:
    if os.path.exists(ds_file):
        print('OK: %s'%ds_file)
    else:
        print('NG: %s'%ds_file)
        check_file+=1
if check_file>0:
    print('Any input ds files is NG')
    sys.exit(1)

def readpkl(f):
    if os.path.exists(f):
        with open(f,'rb') as fo:
            image=pickle.load(fo)
        return image
    else:
        return None

def new_area(num, nptype, shapes):
    if len(shapes) == 2:
        image = np.zeros(
                num * np.prod(shapes[1:]),
                dtype=nptype
            ).reshape(
                -1,shapes[1]
            )
    elif len(shapes) == 3:
        image = np.zeros(
                num * np.prod(shapes[1:]),
                dtype=nptype
            ).reshape(
                -1,shapes[1],shapes[2]
            )
    elif len(shapes) == 4:
        image = np.zeros(
                num * np.prod(shapes[1:]),
                dtype=nptype
            ).reshape(
                -1,shapes[1],shapes[2],shapes[3]
            )
    elif len(shapes) == 1:
        image = np.zeros(
                num,
                dtype=nptype
            )
    return image

def path_summary(path):
    summary=dict()
    for p in path:
        root_path = os.path.split(p)[0]
        if fnmatch.fnmatch(root_path,'*lfw*'):root_path='lfw'
        if fnmatch.fnmatch(root_path,'*indoorCVPR*'):root_path='indoorCVPR'
        if root_path in summary.keys():
            summary[root_path]+=1
        else:
            summary[root_path]=1
    return summary

def shuffle(image, truth, path):
    #np.random.seed(0)
    #np.random.seed(22222)
    idx = np.random.permutation(len(image))
    return image[idx], truth[idx], path[idx]

TRUTH_CENTER = False
if args.truth_center:TRUTH_CENTER=True

TRUTH_MANHAT = False
if args.truth_manhat:TRUTH_MANHAT=TRUTH_CENTER=True

# PASS:1
print('\n# PASS:1')
image_posiN=0
image_negaN=0
image_ambiN=0
for f in files:
    image = readpkl(f)
    if image is None:
        print('Warning: %s not found, skip.'%f)
        continue
    else:
        for k in image.keys():
            if str(k) == 'image_posi':
                image_posiN+=len(image[k])
            elif str(k) == 'image_nega':
                image_negaN+=len(image[k])
            elif k == 'image_ambi':
                image_ambiN+=len(image[k])
            elif k == 'truth_posi':
                truth_posi_shape=image[k].shape
        print('read posi/nega/ambi = %10d/%10d/%10d images after %s file'%(image_posiN,image_negaN,image_ambiN,f))
print('posiN/negaN/ambiN = %d/%d/%d'%(image_posiN,image_negaN,image_ambiN))

print('\n# SETUP TOTAL AREA')
image_posi = new_area(image_posiN, np.uint8, image['image_posi'].shape)
print 'image_posi.shape',image_posi.shape

image_nega = new_area(image_negaN, np.uint8, image['image_nega'].shape)
print 'image_nega.shape',image_nega.shape

image_ambi = new_area(image_ambiN, np.uint8, image['image_ambi'].shape)
print 'image_ambi.shape',image_ambi.shape

truth_posi = new_area(image_posiN, np.float32, truth_posi_shape)
print 'truth_posi.shape',truth_posi.shape

truth_nega = new_area(image_negaN, np.float32, truth_posi_shape)
print 'truth_nega.shape',truth_nega.shape

truth_ambi = new_area(image_ambiN, np.float32, truth_posi_shape)
print 'truth_ambi.shape',truth_ambi.shape

path_posi = new_area(image_posiN, np.dtype('U256'), (0,))
print 'path_posi.shape',path_posi.shape

path_nega = new_area(image_negaN, np.dtype('U256'), (0,))
print 'path_nega.shape',path_nega.shape

path_ambi = new_area(image_ambiN, np.dtype('U256'), (0,))
print 'path_ambi.shape',path_ambi.shape

print('\n# PASS:2')
image_posiN=0
image_negaN=0
image_ambiN=0
truth_posiN=0
truth_negaN=0
truth_ambiN=0
path_posiN =0
path_negaN =0
path_ambiN =0
for f in files:
    image = readpkl(f)
    if image is None:
        print('Warning: %s not found, skip.'%f)
        continue
    else:
        print('->stacking %s.'%f)
        for k in image.keys():
            if str(k) == 'image_posi':
                num=len(image[k])
                image_posi[image_posiN:image_posiN+num]=image[k].copy()
                image_posiN+=num
            elif str(k) == 'image_nega':
                num=len(image[k])
                image_nega[image_negaN:image_negaN+num]=image[k].copy()
                image_negaN+=num
            elif k == 'image_ambi':
                num=len(image[k])
                image_ambi[image_ambiN:image_ambiN+num]=image[k].copy()
                image_ambiN+=num
            elif k == 'truth_posi':
                num=len(image[k])
                truth_posi[truth_posiN:truth_posiN+num]=image[k].copy()
                truth_posiN+=num
            elif k == 'truth_nega':
                num=len(image[k])
                truth_nega[truth_negaN:truth_negaN+num]=image[k].copy()
                truth_negaN+=num
            elif k == 'truth_ambi':
                num=len(image[k])
                truth_ambi[truth_ambiN:truth_ambiN+num]=image[k].copy()
                truth_ambiN+=num
            elif k == 'path_posi':
                num=len(image[k])
                path_posi[path_posiN:path_posiN+num]=image[k].copy()
                path_posiN+=num
            elif k == 'path_nega':
                num=len(image[k])
                path_nega[path_negaN:path_negaN+num]=image[k].copy()
                path_negaN+=num
            elif k == 'path_ambi':
                num=len(image[k])
                path_ambi[path_ambiN:path_ambiN+num]=image[k].copy()
                path_ambiN+=num

# SHUFFLE
if args.no_shuffle is True:
    print('\n# SHUFFLE')
    image_posi, truth_posi, path_posi = shuffle(image_posi,truth_posi,path_posi)
    image_nega, truth_nega, path_nega = shuffle(image_nega,truth_nega,path_nega)
    image_ambi, truth_ambi, path_ambi = shuffle(image_ambi,truth_ambi,path_ambi)
else:
    print('\n# PERMUTATION BY NO-SHUFFLE OPETION')

print('image posiN/negaN/ambiN = %d/%d/%d'%(image_posiN,image_negaN,image_ambiN))
print('truth posiN/negaN/ambiN = %d/%d/%d'%(truth_posiN,truth_negaN,truth_ambiN))
print('path  posiN/negaN/ambiN = %d/%d/%d'%(path_posiN ,path_negaN ,path_ambiN))

print('\n# STATISTICS NEGA/POSI TRUTH BY 1.0/0.0')
truth_posi_zeros = len(truth_posi[truth_posi==0.])
truth_nega_zeros = len(truth_nega[truth_nega==0.])
truth_ambi_zeros = len(truth_ambi[truth_ambi==0.])
truth_posi_nonzs = np.count_nonzero(truth_posi)
truth_nega_nonzs = np.count_nonzero(truth_nega)
truth_ambi_nonzs = np.count_nonzero(truth_ambi)
print('truth posi 1.0/0.0=%8d/%8d'%(truth_posi_nonzs,truth_posi_zeros))
print('truth nega 1.0/0.0=%8d/%8d'%(truth_nega_nonzs,truth_nega_zeros))
print('truth ambi 1.0/0.0=%8d/%8d'%(truth_ambi_nonzs,truth_ambi_zeros))

using_posi = image_posiN
using_nega = int((truth_posi_nonzs / truth_posi.shape[1] - truth_posi_zeros / truth_posi.shape[1]))
if TRUTH_CENTER or TRUTH_MANHAT:
    using_nega = image_negaN
print('Usable leaning images is using_posi/using_nega = %d/%d'%(using_posi,using_nega))

using_posi_test = int(using_posi/10)
using_nega_test = int(using_nega/10)
using_posi_train= using_posi - using_posi_test
using_nega_train= using_nega - using_nega_test
print('Separate for train posi/nega = %d/%d'%(using_posi_train,using_nega_train))
print('Separate for test  posi/nega = %d/%d'%(using_posi_test ,using_nega_test))
using_trainN = using_posi_train + using_nega_train
using_testN  = using_posi_test  + using_nega_test
print('\n# Finally images train/test = %d/%d'%(using_trainN,using_testN))

# *[12]_train are np.slice type
# To copy into train_image and train_truth
# size of image posi and truth posi is same, 
#--------------------------------- train image
# image posi [ p1_train p2_test ]
#              |
# train image[ x1_posi  x2_nega ]
#                      /
# image nega [ n1_train n2_test ]
#--------------------------------- train truth
# truth posi [ p1_train p2_test ]
#              |
# train truth[ x1_posi  x2_nega ]
#                      /
# truth nega [ n1_train n2_test ]
#---------------------------------

# To copy into test_image and test_truth
#--------------------------------- test image
# image posi [ p1_train p2_test ]
#                      /
# test  image[  X1_posi X2_nega ]
#                       |
# image nega [ n1_train n2_test ]
#--------------------------------- test truth
# truth posi [ p1_train p2_test ]
#                      /
# test  truth[  X1_posi X2_nega ]
#                       |
# truth nega [ n1_train n2_test ]
#---------------------------------

#
p1_train = np.s_[                0 : using_posi_train ]
p2_test  = np.s_[ using_posi_train : using_posi_train + using_posi_test ]
n1_train = np.s_[                0 : using_nega_train ]
n2_test  = np.s_[ using_nega_train : using_nega_train + using_nega_test ]
#
x1_posi  = np.s_[                0 : using_posi_train ]
x2_nega  = np.s_[ using_posi_train : using_posi_train + using_nega_train]
#
X1_posi  = np.s_[                0 : using_posi_test ]
X2_nega  = np.s_[ using_posi_test  : using_posi_test  + using_nega_test]

print('\n# RECONSTRUCTURE PROCESS BY BELOW SLICE..')
print 'p1_train:' ,p1_train
print 'p2_test :' ,p2_test
print 'n1_train:' ,n1_train
print 'n2_test :' ,n2_test
print 'x1_posi :' ,x1_posi
print 'x2_nega :' ,x2_nega
print 'X1_posi :' ,X1_posi
print 'X2_nega :' ,X2_nega

# total areas
train_image = new_area(using_trainN, np.uint8,   image_posi.shape)
train_truth = new_area(using_trainN, np.float32, truth_posi.shape)
test_image  = new_area(using_testN , np.uint8,   image_posi.shape)
test_truth  = new_area(using_testN , np.float32, truth_posi.shape)
train_path  = new_area(using_trainN, np.dtype('U256'), (0,))
test_path   = new_area(using_testN , np.dtype('U256'), (0,))

# Coping
train_image[x1_posi] = image_posi[p1_train]
train_image[x2_nega] = image_nega[n1_train]
train_truth[x1_posi] = truth_posi[p1_train]
train_truth[x2_nega] = truth_nega[n1_train]
train_path[x1_posi]  = path_posi[p1_train]
train_path[x2_nega]  = path_nega[n1_train]

test_image[X1_posi]  = image_posi[p2_test]
test_image[X2_nega]  = image_nega[n2_test]
test_truth[X1_posi]  = truth_posi[p2_test]
test_truth[X2_nega]  = truth_nega[n2_test]
test_path[X1_posi]   = path_posi[p2_test]
test_path[X2_nega]   = path_nega[n2_test]

print 'train_image.shape :', train_image.shape
print 'train_truth.shape :', train_truth.shape
print 'test_image.shape  :', test_image.shape
print 'test_truth.shape  :', test_truth.shape

print('\n# STATISTICS')
train_nonz = np.count_nonzero(train_truth)
train_alls = np.prod(train_truth.shape)
nonz_ratio = float(train_nonz)/float(train_alls)
print('train data box-wise nonzero/all ratio = %d/%d = %5.2f'%(train_nonz, train_alls, 100.*nonz_ratio))
test_nonz = np.count_nonzero(test_truth)
test_alls = np.prod(test_truth.shape)
nonz_ratio = float(test_nonz)/float(test_alls)
print('test  data box-wise nonzero/all ratio = %d/%d = %5.2f'%(test_nonz, test_alls, 100.*nonz_ratio))

train_summary = path_summary(train_path)
for k in train_summary.keys():
    if train_summary[k]>10:
        print('train images/path = %10d :%s'%(train_summary[k],k))

test_summary = path_summary(test_path)
for k in test_summary.keys():
    if test_summary[k]>10:
        print('test  images/path = %10d :%s'%(test_summary[k],k))

if not args.only_check:
    print('\n# SAVING')
    with open('image.pkl','wb') as f:
        print('  %s'%'image.pkl')
        image_all = {
            'train': train_image,
            'test':  test_image
            }
        pickle.dump(image_all,f)

    with open('label.pkl','wb') as f:
        print('  %s'%'label.pkl')
        truth_all = {
            'train': train_truth,
            'test':  test_truth
            }
        pickle.dump(truth_all,f)
else:
    print('check only, dont create pkl')

sys.exit(1)
