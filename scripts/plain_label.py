import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import re
import fnmatch
import argparse

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["person", "chair", "sofa", "tvmonitor"]


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
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels2/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def findfile(dirname,pattern='*.jpg'):
    files=[]
    for root,dirs,names in os.walk(dirname):                     
        finded_files=[os.path.join(root,f) for f in names      
            if fnmatch.fnmatch(os.path.join(root,f), pattern)] 
        files.extend(finded_files) 
    print('find files=%d in %s-directory'%(len(files),dirname))
    for f in files[:10]:print f
    return files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Original dataset and annotation')
    parser.add_argument('directory', nargs='+', type=str)
    parser.add_argument('--nega', action="store_true")
    parser.add_argument('--posi', action="store_false")
    args = parser.parse_args()

    sets = args.directory
    print(args.directory)
    wd = getcwd()

    mark='_posi'
    if args.nega:mark='_nega'
    for dirname in sets:
        if not os.path.exists(dirname):
            print('%s not found'%(dirname))
            sys.exit(1)
        image_ids = findfile(dirname)
        print('...')
        list_file = open('%s%s.txt'%(dirname,mark), 'w')
        for image_id in image_ids:
            if len(re.findall(' ',image_id))!=0:continue
            list_file.write('%s/%s\n'%(wd,image_id))
            label_id = re.sub('.jpg','.txt',image_id)
            w=open(label_id,'w')
            if args.nega is False:
                w.write('14 0.50 0.5 0.5 0.5\n')
            w.close()
        list_file.close()
        if args.nega is False:
            print('*POSI* dataset processed, sure correct?')
        else:
            print('*NEGA* dataset processed, sure correct?')
        print('created %s%s.txt file %d-lines'%(dirname,mark,len(image_ids)))

    #os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train2.txt")
    #os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

