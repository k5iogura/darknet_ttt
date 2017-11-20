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
#from loss_custom import *

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["person"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    posN=0
    rejP=0
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    #out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    img = cv2.imread('VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    if TRUTH_MANHAT:
        Gtruth = np.zeros((2,16),dtype=np.float32).reshape(-1,4,4)
    else:
        Gtruth = np.zeros(16,dtype=np.float32).reshape(4,4)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        B=(int(xmlbox.find('xmin').text),int(xmlbox.find('xmax').text),int(xmlbox.find('ymin').text),int(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        xbox = int(((b[0]+b[1])/2.0)/(w/DIVISIONS))
        ybox = int(((b[2]+b[3])/2.0)/(h/DIVISIONS))
        xp = int(round(b[0]/w/DIV_RATE,1))
        xq = int(round(b[1]/w/DIV_RATE,1))
        yp = int(round(b[2]/h/DIV_RATE,1))
        yq = int(round(b[3]/h/DIV_RATE,1))
        if (b[1]-b[0])>=MIN_PATCH and (b[3]-b[2])>=MIN_PATCH:
            posN+=1
            if DEBUG1:print("%s %d %d %d %d : %d %d %d %d : %d %d"%(image_id,xp,xq,yp,yq,b[0],b[1],b[2],b[3],w,h))
            if TRUTH_CENTER:
                if TRUTH_MANHAT:
                    mR = int(DIVISIONS*min(bb[2],bb[3]))+1
                    Gtruth[1][ybox][xbox] = mR/float(DIVISIONS)
                    Gtruth[0][ybox][xbox] = 1.
                else:
                    Gtruth[ybox][xbox] = 1.
            else:
                for j in range(yp,yq+1):
                    for i in range(xp,xq+1):
                        Gtruth[j][i]=1.
        else:
            rejP+=1
        if DEBUG1:cv2.rectangle(img,(B[0],B[2]),(B[1],B[3]),(255,0,255),8)
    # write out into check file for debug
    #for j in range(0,DIVISIONS):
    #    for i in range(0,DIVISIONS):
    #        out_file.write(str("%2d "%(Gtruth[i][j])))
    #out_file.write("\n")
    # swap RGB Pixel-Wise to BGR Channel-Wise and save global area
    img_pw_rgb = cv2.resize(img,(NN_IN_SIZE,NN_IN_SIZE))
    img_cw_bgr0=img_pw_rgb.transpose(2,0,1).copy()          # reform HWC to CHW
    img_cw_bgr1=img_pw_rgb.transpose(2,0,1).copy()          # reform HWC to CHW
    img_cw_bgr0[0]=img_cw_bgr1[2]
    img_cw_bgr0[2]=img_cw_bgr1[0]
    # save global area
    if TRUTH_MANHAT:
        truth  = Gtruth.reshape(2,DIVISIONS*DIVISIONS)
    else:
        truth  = Gtruth.reshape(DIVISIONS*DIVISIONS)
    #if DEBUG1:img = cv2.resize(img,(w,h))
    if DEBUG1:img = cv2.resize(img_cw_bgr0.transpose(1,2,0),(w,h))  # reform CHW to HWC for Debug
    if DEBUG1:cv2.imshow('%s'%(image_id),img)
    if DEBUG1:
        while(1):
            key=cv2.waitKey(100)
            if key == 27: sys.exit(1)   # ESC
            if key == 32: break         # space
    if DEBUG1:cv2.destroyAllWindows()
    in_file.close()
    #out_file.close()
    return posN, rejP, img_cw_bgr0, truth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Original dataset and annotation into pickle')
    parser.add_argument('--debug1', action="store_true")
    parser.add_argument('--debug2', action="store_true")
    parser.add_argument('--image_file', '-i', type=str, default="voc_ds.pkl")
    parser.add_argument('--max_count',  '-m', type=int, default=30000)
    parser.add_argument('--nn_in_size',       type=int, default=32)
    parser.add_argument('--min_patch',        type=int, default=128)
    parser.add_argument('--truth_center','-tc',action="store_true")
    parser.add_argument('--truth_manhat','-tm',action="store_true")
    args = parser.parse_args()

    voc_image_file = args.image_file
    DEBUG1=False
    if args.debug1:DEBUG1=True
    DEBUG2=False
    if args.debug2:DEBUG2=True

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

    img_count=0
    max_count=200
    if args.max_count: max_count=int(args.max_count)
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
    path_posi = np.zeros(max_count,dtype=np.dtype('U256'))
    path_nega = path_posi.copy()
    path_ambi = path_posi.copy()

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

    counter=0
    image_posiN=0
    image_negaN=0
    image_ambiN=0
    for year, image_set in sets:
        if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
            os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
        image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        list_file = open('%s_%s.txt'%(year, image_set), 'w')
        for image_id in image_ids:
            file_name = '%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id)
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
            posN, rejP, image, truth = convert_annotation(year, image_id)
            if posN>0 and posN!=rejP:
                image_posi[image_posiN] = image.copy()
                if TRUTH_MANHAT:
                    truth_posi[:][image_posiN] = truth.copy()
                else:
                    truth_posi[image_posiN] = truth.copy()
                path_posi[image_posiN]  = file_name
                image_posiN+=1
            elif posN==0 and rejP==0:
                image_nega[image_negaN] = image.copy()
                if TRUTH_MANHAT:
                    truth_nega[:][image_negaN] = truth.copy()
                else:
                    truth_nega[image_negaN] = truth.copy()
                path_nega[image_negaN]  = file_name
                image_negaN+=1
            else:
                image_ambi[image_ambiN] = image.copy()
                if TRUTH_MANHAT:
                    truth_ambi[:][image_ambiN] = truth.copy()
                else:
                    truth_ambi[image_ambiN] = truth.copy()
                path_ambi[image_ambiN] = file_name
                image_ambiN+=1
            counter+=1
            if counter%200 == 0:print(
                "Processing %d/%d(posi/nega/rej=%d/%d/%d)"%(counter,max_count,image_posiN,image_negaN,image_ambiN)
                )
            if image_posiN+image_negaN+image_ambiN>=max_count:break
        list_file.close()

    print("*Dataset statistics*")
    print("all : image_posiN/image_negaN=%d/%d"%(image_posiN,image_negaN))
    # number of data abount posi vs nega
    #posi:nega=7:3
    #3posi=7nega
    #nega=3posi/7
#    image_negaN_tmp = int(3*image_posiN/7)
#    if image_negaN_tmp>image_negaN:
#        print("Error:Negative Data is Shortage. %d data but %d Needed"%(image_negaN,image_negaN_tmp))
#    else:
#        image_negaN = image_negaN_tmp

    # split image_posiN into train and test
#    test_posiN = int(image_posiN / 10)
#    train_posiN= image_posiN - test_posiN

    # split image_negaN into train and test
#    test_negaN = int(image_negaN / 10)
#    train_negaN= image_negaN - test_negaN
#    print("7:3 : image_posiN/image_negaN=%d/%d"%(image_posiN,image_negaN))

#    train_image[:train_posiN]                        = image_posi[:train_posiN]
#    train_image[train_posiN:train_posiN+train_negaN] = image_nega[:train_negaN]
#    train_prob[:train_posiN]                         = truth_posi[:train_posiN]
#    train_prob[train_posiN:train_posiN+train_negaN]  = truth_nega[:train_negaN]

#    test_image[:test_posiN]                          = image_posi[train_posiN:train_posiN+test_posiN]
#    test_image[test_posiN:test_posiN+test_negaN]     = image_nega[train_negaN:train_negaN+test_negaN]
#    test_prob[:test_posiN]                           = truth_posi[train_posiN:train_posiN+test_posiN]
#    test_prob[test_posiN:test_posiN+test_negaN]      = truth_nega[train_negaN:train_negaN+test_negaN]
#    print("train = %06d = posi/nega= %05d/%05d"%(train_posiN+train_negaN,train_posiN,train_negaN))
#    print("test  = %06d = posi/nega= %05d/%05d"%(test_posiN+test_negaN,test_posiN,test_negaN))

    if counter > 0:

        if TRUTH_MANHAT:
            image_buf = {
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
            image_buf = {
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
        # write global area out
        with open(voc_image_file,'wb') as f:
            pickle.dump(image_buf,f)

        # write list out
        #os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
        #os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

    else:
        print("Error:No Data")

    if DEBUG2:
        for i in range(0,5):
            S=args.nn_in_size
            posi=cv2.resize(image_buf['image_posi'][i].transpose(2,0,1).astype(np.uint8),(S,S))
            nega=cv2.resize(image_buf['image_nega'][i].transpose(2,0,1).astype(np.uint8),(S,S))
            ambi=cv2.resize(image_buf['image_ambi'][i].transpose(2,0,1).astype(np.uint8),(S,S))
            cv2.imshow('return image posi', posi)
            cv2.imshow('return image nega', nega)
            cv2.imshow('return image ambi', ambi)
            while(1):
                key = cv2.waitKey(30)
                if key ==32:break
                if key ==27:sys.exit(1)

