import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np
import cv2
import sys
from pdb import *

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

DIVISIONS=4
DIV_RATE=(1./DIVISIONS+0.01)
NN_IN_SIZE=32
NN_IN_CHNL=3
MIN_PATCH=128

img_count=0
max_count=3000
image_nd = np.zeros(
        max_count * NN_IN_SIZE * NN_IN_SIZE * NN_IN_CHNL,
        dtype=np.float32
    ).reshape(
        max_count,
        NN_IN_CHNL,
        NN_IN_SIZE,
        NN_IN_SIZE
    )

prob_nd = np.zeros(
        max_count * DIVISIONS * DIVISIONS,
        dtype=np.int32
    ).reshape(
        max_count,
        DIVISIONS * DIVISIONS
    )

DEBUG=True
DEBUG=False
def convert_annotation(year, image_id):
    global img_count
    global image_nd
    global prob_nd
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    img = cv2.imread('VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

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
        xp = int(round(b[0]/w/DIV_RATE,1))
        xq = int(round(b[1]/w/DIV_RATE,1))
        yp = int(round(b[2]/h/DIV_RATE,1))
        yq = int(round(b[3]/h/DIV_RATE,1))
        if (b[1]-b[0])>=MIN_PATCH or (b[3]-b[2])>=MIN_PATCH:
            if DEBUG:print("%s %d %d %d %d : %d %d %d %d : %d %d"%(image_id,xp,xq,yp,yq,b[0],b[1],b[2],b[3],w,h))
            for j in range(yp,yq+1):
                for i in range(xp,xq+1):
                    Gtruth[i][j]=1.
                    if DEBUG:sys.stdout.write("%2d-"%(j*DIVISIONS+i))
        if DEBUG:cv2.rectangle(img,(B[0],B[2]),(B[1],B[3]),(255,0,255),8)
    # write out into check file for debug
    for j in range(0,DIVISIONS):
        for i in range(0,DIVISIONS):
            out_file.write(str("%2d "%(Gtruth[i][j])))
    out_file.write("\n")
    # swap RGB Pixel-Wise to BGR Channel-Wise and save global area
    img_pw_rgb = cv2.resize(img,(NN_IN_SIZE,NN_IN_SIZE))
    img_cw_bgr0=img_pw_rgb.transpose(2,1,0).copy()
    img_cw_bgr1=img_pw_rgb.transpose(2,1,0).copy()
    img_cw_bgr0[0]=img_cw_bgr1[2]
    img_cw_bgr0[2]=img_cw_bgr1[0]
    image_nd[img_count] = img_cw_bgr1
    # save global area
    prob_nd[img_count]=Gtruth.reshape(DIVISIONS*DIVISIONS)
    img_count+=1
    if DEBUG:img = cv2.resize(img,(w,h))
    if DEBUG:cv2.imshow('%s'%(image_id),img)
    if DEBUG:
        while(1):
            if cv2.waitKey(100) == 27: break
    if DEBUG:cv2.destroyAllWindows()
    in_file.close()
    out_file.close()

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    #stopcnt = 0
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
	#stopcnt+=1
	#if stopcnt >= 100: break
    list_file.close()

# write global area out
print("img_count=%d"%img_count)
with open('image.pkl','wb') as f:
    pickle.dump(image_nd[:img_count],f)
with open('prob.pkl','wb') as f:
    pickle.dump(prob_nd[:img_count],f)

# write list out
os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

