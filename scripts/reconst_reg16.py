import pickle
import numpy as np
import os
import sys
import argparse

files = ['voc_ds.pkl', 'lfw_ds.pkl', 'indoorCVPR_09_ds.pkl']

parser = argparse.ArgumentParser(description='check Original dataset and annotation into pickle')
parser.add_argument('--ds_file', '-d', type=str, default='voc_ds.pkl')
args = parser.parse_args()


def readpkl(f):
    if os.path.exists(f):
        with open(f,'rb') as fo:
            print("reading:%s"%f)
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
    elif len(shapes) == 4:
        image = np.zeros(
                num * np.prod(shapes[1:]),
                dtype=nptype
            ).reshape(
                -1,shapes[1],shapes[2],shapes[3]
            )
    return image

# PASS:1
print('# PASS:1')
image_posiN=0
image_negaN=0
image_ambiN=0
for f in files:
    image = readpkl(f)
    if image is None:
        print('Warning: %s not found, skip.'%f)
        continue
    else:
        print('->analizing %s.'%f)
        for k in image.keys():
            if str(k) == 'image_posi':
                image_posiN+=len(image[k])
            elif str(k) == 'image_nega':
                image_negaN+=len(image[k])
            elif k == 'image_ambi':
                image_ambiN+=len(image[k])
            elif k == 'truth_posi':
                truth_posi_shape=image[k].shape
print('posiN/negaN/ambiN = %d/%d/%d'%(image_posiN,image_negaN,image_ambiN))

print('# MAKE TOTAL AREA')
image_posi = new_area(image_posiN, np.uint8, image['image_posi'].shape)
print 'image_posi.shape',image_posi.shape

image_nega = new_area(image_negaN, np.uint8, image['image_nega'].shape)
print 'image_nega.shape',image_nega.shape

image_ambi = new_area(image_ambiN, np.uint8, image['image_ambi'].shape)
print 'image_ambi.shape',image_ambi.shape

truth_posi = new_area(image_posiN, np.int32, truth_posi_shape)
print 'truth_posi.shape',truth_posi.shape

truth_nega = new_area(image_negaN, np.int32, truth_posi_shape)
print 'truth_nega.shape',truth_nega.shape

truth_ambi = new_area(image_ambiN, np.int32, truth_posi_shape)
print 'truth_ambi.shape',truth_ambi.shape

print('# PASS:2')
image_posiN=0
image_negaN=0
image_ambiN=0
truth_posiN=0
truth_negaN=0
truth_ambiN=0
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

print('image posiN/negaN/ambiN = %d/%d/%d'%(image_posiN,image_negaN,image_ambiN))
print('truth posiN/negaN/ambiN = %d/%d/%d'%(truth_posiN,truth_negaN,truth_ambiN))

print('# CALCURATE NEGA/POSI IMAGES')
truth_posi_zeros = len(truth_posi[truth_posi==0.])
truth_nega_zeros = len(truth_nega[truth_nega==0.])
truth_ambi_zeros = len(truth_ambi[truth_ambi==0.])
truth_posi_nonzs = np.prod(truth_posi.shape) - truth_posi_zeros
truth_nega_nonzs = np.prod(truth_nega.shape) - truth_nega_zeros
truth_ambi_nonzs = np.prod(truth_ambi.shape) - truth_ambi_zeros
print('truth posi 1.0/0.0=%8d/%8d'%(truth_posi_nonzs,truth_posi_zeros))
print('truth nega 1.0/0.0=%8d/%8d'%(truth_nega_nonzs,truth_nega_zeros))
print('truth ambi 1.0/0.0=%8d/%8d'%(truth_ambi_nonzs,truth_ambi_zeros))

using_posi = image_posiN
using_nega = int(truth_posi_nonzs / truth_posi.shape[1])
print('Usable leaning images is using_posi/using_nega = %d/%d'%(using_posi,using_nega))

using_posi_test = int(using_posi/10)
using_nega_test = int(using_nega/10)
using_posi_train= using_posi - using_posi_test
using_nega_train= using_nega - using_nega_test
print('Separate for train posi/nega = %d/%d'%(using_posi_train,using_nega_train))
print('Separate for test  posi/nega = %d/%d'%(using_posi_test ,using_nega_test))
using_trainN = using_posi_train + using_nega_train
using_testN  = using_posi_test  + using_nega_test
print('Finally images train/test = %d/%d'%(using_trainN,using_testN))

sys.exit(1)
print('# SAVING')
with open('image.pkl','wb') as f:
    print('  %s','image.pkl')
    image_all = {
        'train': image_posi[:using_posi],
        'test':  image_posi[:using_posi]
        }
    pickle.dump(image_buf,f)

