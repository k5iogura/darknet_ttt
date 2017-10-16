import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np
import cv2
import sys
import argparse
from pdb import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Original dataset and annotation into pickle')
    parser.add_argument('dataset_prefix', type=str)
    parser.add_argument('--debug1', action="store_true")
    parser.add_argument('--debug2', action="store_true")
    parser.add_argument('--image_file', '-i', type=str, default="image.pkl")
    parser.add_argument('--prob_file',  '-p', type=str, default="prob.pkl")
    parser.add_argument('--max_count',  '-m', type=int, default=0)
    parser.add_argument('--nn_in_size',       type=int, default=32)
    parser.add_argument('--min_patch',        type=int, default=128)
    args = parser.parse_args()

    pkl_image_file = '%s_%s'%(args.dataset_prefix,args.image_file)
    pkl_prob_file  = '%s_%s'%(args.dataset_prefix,args.prob_file)
    print('create %s and %s'%(pkl_image_file,pkl_prob_file))

    DEBUG1=False
    if args.debug1:DEBUG1=True

    DIVISIONS=4
    DIV_RATE=(1./DIVISIONS+0.01)
    NN_IN_SIZE=32
    NN_IN_CHNL=3
    MIN_PATCH=128
    if args.nn_in_size:NN_IN_SIZE=int(args.nn_in_size)
    if args.min_patch:MIN_PATCH=int(args.min_patch)

    image_posiN=0
    image_negaN=0
    files=[]
    max_count_tmp=0
    if os.path.exists(args.dataset_prefix):
        jpg_dirs  = [name_dir for name_dir in os.listdir(args.dataset_prefix)]
        for jpg_dir in jpg_dirs:
            jpg_files = os.listdir(args.dataset_prefix+'/'+str(jpg_dir))
            for jpg_file in jpg_files:
                if args.max_count>0 and args.max_count <= max_count_tmp: break
                files.append(args.dataset_prefix+'/'+str(jpg_dir)+'/'+str(jpg_file))
                max_count_tmp+=1
        max_count=len(files)
        print('In %s dir, jpg files = %d'%(args.dataset_prefix,len(files)))
    else:
        print("%s directory not found"%(args.dataset_prefix))
        sys.exit(1)

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
    train_image= image_posi.copy()
    test_image = image_posi.copy()

    prob_posi = np.zeros(
            max_count * DIVISIONS * DIVISIONS,
            dtype=np.int32
        ).reshape(
            max_count,
            DIVISIONS * DIVISIONS
        )
    prob_nega = prob_posi.copy()
    train_prob= prob_posi.copy()
    test_prob = prob_posi.copy()

    wd = getcwd()

    counter=0
    prob_const  = np.array([0,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1],dtype=np.float32)
    for jpg in files:
        img = cv2.imread(jpg)
        img = cv2.resize(img,(NN_IN_SIZE,NN_IN_SIZE))
        img_bgr_cw0=img.transpose(2,1,0).copy()
        img_rgb_cw1=img.transpose(2,1,0).copy()
        img_bgr_cw0[0] = img_rgb_cw1[2]
        img_bgr_cw0[2] = img_rgb_cw1[0]
        image_posi[image_posiN] = img_bgr_cw0
        prob_posi[image_posiN]  = prob_const
        image_posiN+=1
        counter+=1
        if counter%200 == 0:print("Processing %d/%d"%(counter,max_count))

    # write pkl file out
    if image_posiN > 0:

        # write global area out
        with open(pkl_image_file,'wb') as f:
            pickle.dump(image_posi,f)
        with open(pkl_prob_file,'wb') as f:
            pickle.dump(prob_posi,f)

        # write list out
        #os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
        #os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

    else:
        print("Error:No Data")

    if DEBUG1:
        for i in range(10,15):
            cv2.imshow('return image', image_posi[i].transpose(2,1,0).astype(np.uint8))
            print(prob_posi[i])
            while(1):
                key = cv2.waitKey(30)
                if key ==32:break       # space
                if key ==27:sys.exit(1) # esc

