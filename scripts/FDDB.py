import os
import fnmatch
import numpy as xp
import cv2
import re
import sys
from pdb import *

def trans_annotation(jpg_listfile,dirname,filename):
    human = 14
    CWD = os.getcwd()
    with open(filename) as f:
        lines = [l.strip() for l in f.readlines()]
    jpg_list = open(jpg_listfile,'a')
    for i,line in enumerate(lines):
        if len(re.findall('/',line))>0:
            imagename = dirname+'/'+line+'.jpg'
            labelname = dirname+'/'+line+'.txt'
            jpg_list.write('%s/%s\n'%(CWD,imagename))
            if os.path.exists(imagename) is False: continue
            if os.path.exists(labelname) is True:
                with open(labelname,'w') as l:pass
            image = cv2.imread(imagename)
            (H,W,C)=image.shape
        elif len(line) == 1:
            faceN = int(line)
        elif len(line.split()) == 6:
            # Annotation FDDB : <major_axis_radius> <minor_axis_radius> <angle> <center_x> <center_y> 1>
            # Annotation Dark : <object-class> <x> <y> <width> <height>
            annotation = line.split()
            r=float(min(annotation[1],annotation[0]))/H
            x=float(annotation[3])/W
            y=float(annotation[4])/H
            #x,y,w,h = (x-r, y+r, 2*r, 2*r)  # Dark format
            x,y,w,h = (x, y, 2*r, 2*r)      # My format
            if x<0.:x=0.
            if x>=1.:x=0.999
            if y<0.:y=0.
            if y>=1.:y=0.999
            if w>1:w=1.
            if h>1:h=1.
            assert x>=0 and y<1. and w>0. and h>0.
            print('%s: %d %.3f %.3f %.3f %.3f'%(labelname,human,x,y,r,r))
            with open(labelname,'a') as l:
                l.write('%d %.3f %.3f %.3f %.3f\n'%(human,x,y,r,r))
        else:
            continue
    jpg_list.close()

if __name__ == '__main__':
    dirname ='FDDB-folds'
    #filename='FDDB-fold-01-ellipseList.txt'

    #trans_annotation(dirname,filename)

    FDDB_jpg_listfile = 'FDDB_jpg.txt'
    files=[]
    for root,dirs,names in os.walk(dirname):                     
        finded_files=[os.path.join(root,f) for f in names      
            if fnmatch.fnmatch(os.path.join(root,f), '*ellipse*.txt')] 
        files.extend(finded_files)
    with open(FDDB_jpg_listfile,'w') as jpg_list: jpg_list.close()
    for filename in files:
        print(filename)
        trans_annotation(FDDB_jpg_listfile, dirname, filename) 

