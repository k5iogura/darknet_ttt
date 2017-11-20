import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np
import cv2
import sys
import argparse
import fnmatch
from pdb import *
from loss_custom import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Original dataset and annotation into pickle')
    parser.add_argument('dataset_prefix', type=str)
    parser.add_argument('--debug1', action="store_true")
    parser.add_argument('--debug2', action="store_true")
    parser.add_argument('--image_file', '-i', type=str, default="ds.pkl")
    parser.add_argument('--nega',       '-n', action="store_true")
    parser.add_argument('--max_count',  '-m', type=int, default=0)
    parser.add_argument('--nn_in_size',       type=int, default=32)
    parser.add_argument('--min_patch',        type=int, default=128)
    parser.add_argument('--truth_center','-tc',action="store_true")
    parser.add_argument('--truth_manhat','-tm',action="store_true")
    args = parser.parse_args()

    pkl_image_file = '%s_%s'%(args.dataset_prefix,args.image_file)
    print('create %s'%(pkl_image_file))

    DEBUG1=False
    if args.debug1:DEBUG1=True

    TRUTH_CENTER = False
    if args.truth_center:TRUTH_CENTER=True

    TRUTH_MANHAT = False
    if args.truth_manhat:TRUTH_CENTER=TRUTH_MANHAT=True

    DIVISIONS=4
    DIV_RATE=(1./DIVISIONS+0.01)
    NN_IN_SIZE=32
    NN_IN_CHNL=3
    MIN_PATCH=128
    if args.nn_in_size:NN_IN_SIZE=int(args.nn_in_size)
    if args.min_patch:MIN_PATCH=int(args.min_patch)

    files=[]
    max_count_tmp=0
    for root,dirs,names in os.walk(args.dataset_prefix):
        finded_files = [os.path.join(root,f) for f in names
            if fnmatch.fnmatch(os.path.join(root,f),'*.jpg')]
        files.extend(finded_files)
        max_count_tmp+=len(finded_files)
        if args.max_count>0 and args.max_count <= max_count_tmp: break

    max_count=len(files)
    print('In %s dir, jpg files = %d'%(args.dataset_prefix,len(files)))
    if max_count==0: sys.exit(1)

    image_posi = np.zeros(
            max_count * NN_IN_SIZE * NN_IN_SIZE * NN_IN_CHNL,
            dtype=np.uint8
        ).reshape(
            max_count,
            NN_IN_CHNL,
            NN_IN_SIZE,
            NN_IN_SIZE
        )
    image_nega = image_posi.copy()
    image_ambi = image_posi.copy()
    path_posi  = np.zeros(max_count,dtype=np.dtype('U256'))
    path_nega  = path_posi.copy()
    path_ambi  = path_posi.copy()

    if TRUTH_MANHAT:
        truth_posi = np.zeros(
                2 * max_count * DIVISIONS * DIVISIONS,
                dtype=np.float32
            ).reshape(
                max_count,
                2,
                DIVISIONS * DIVISIONS
            )
    else:
        truth_posi = np.zeros(
                max_count * DIVISIONS * DIVISIONS,
                dtype=np.int32
            ).reshape(
                max_count,
                DIVISIONS * DIVISIONS
            )
    truth_nega = truth_posi.copy()
    truth_ambi = truth_posi.copy()

    wd = getcwd()

    image_posiN=0
    image_negaN=0
    image_ambiN=0
    counter=0
    if args.nega:
        if TRUTH_CENTER:
            if TRUTH_MANHAT:
                truth_const  = np.array(
                    [
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    ],    dtype=np.float32
                )
            else:
                truth_const  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
        else:
            truth_const  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
    else:
        if TRUTH_CENTER:
            if TRUTH_MANHAT:
                truth_const  = np.array(
                    [
                        [0,0,0,0,0,0,0,0,0,1.0,1.0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0.5,0.5,0,0,0,0,0]
                    ],    dtype=np.float32
                )
            else:
                truth_const  = np.array([0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],dtype=np.float32)
        else:
            truth_const  = np.array([0,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1],dtype=np.float32)
            truth_const  = np.array([0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0],dtype=np.float32)
    for jpg in files:
        img = cv2.imread(jpg)
        if img is None: continue
        img = cv2.resize(img,(NN_IN_SIZE,NN_IN_SIZE))
        img_bgr_cw0=img.transpose(2,0,1).copy()         # reform HWC to CHW from pixelwise to RGB wise
        img_rgb_cw1=img.transpose(2,0,1).copy()         # reform HWC to CHW
        img_bgr_cw0[0] = img_rgb_cw1[2]
        img_bgr_cw0[2] = img_rgb_cw1[0]
        if args.nega is False:
            image_posi[image_posiN] = img_bgr_cw0
            if TRUTH_MANHAT:
                truth_posi[:][image_posiN]  = truth_const
            else:
                truth_posi[image_posiN]  = truth_const
            path_posi[image_posiN]  = jpg
            image_posiN+=1
        else:
            image_nega[image_negaN] = img_bgr_cw0
            if TRUTH_MANHAT:
                truth_nega[:][image_negaN]  = truth_const
            else:
                truth_nega[image_negaN]  = truth_const
            path_nega[image_negaN]  = jpg
            image_negaN+=1
        counter+=1
        if counter%200 == 0:print("Processing %d/%d"%(counter,max_count))

    # write pkl file out
    if counter > 0:

        # write global area out
        if TRUTH_MANHAT:
            image_buf={
                'image_posi':image_posi[:image_posiN],
                'image_nega':image_nega[:image_negaN],
                'image_ambi':image_ambi[:image_ambiN],
                'truth_posi':truth_posi[:][:image_posiN],
                'truth_nega':truth_nega[:][:image_negaN],
                'truth_ambi':truth_ambi[:][:image_ambiN],
                'path_posi':path_posi[:image_posiN],
                'path_nega':path_nega[:image_negaN],
                'path_ambi':path_ambi[:image_ambiN]
                }
        else:
            image_buf={
                'image_posi':image_posi[:image_posiN],
                'image_nega':image_nega[:image_negaN],
                'image_ambi':image_ambi[:image_ambiN],
                'truth_posi':truth_posi[:image_posiN],
                'truth_nega':truth_nega[:image_negaN],
                'truth_ambi':truth_ambi[:image_ambiN],
                'path_posi':path_posi[:image_posiN],
                'path_nega':path_nega[:image_negaN],
                'path_ambi':path_ambi[:image_ambiN]
                }
        with open(pkl_image_file,'wb') as f:
            pickle.dump(image_buf,f)
        print('%s include %d image files (posi/nega/ambi=%d/%d/%d)'%(pkl_image_file,counter,image_posiN,image_negaN,image_ambiN))

        # write list out
        #os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
        #os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

    else:
        print("Error:No Data")

    if DEBUG1:
        for i in range(10,15):
            cv2.imshow('return image', image_posi[i].transpose(1,2,0).astype(np.uint8)) # reform CHW to HWC
            #print(truth_posi[i])
            while(1):
                key = cv2.waitKey(30)
                if key ==32:break       # space
                if key ==27:sys.exit(1) # esc

