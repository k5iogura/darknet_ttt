import os
import numpy as xp
import cv2
import re
import sys

with open('FDDB-folds/FDDB-fold-01-ellipseList.txt') as f:
    lines = [l.strip() for l in f.readlines()]

for i,line in enumerate(lines):
    if len(re.findall('/',line))>0:
        imagename = line+'.jpg'
        if os.path.exists(imagename) is False:
            continue
    elif len(line) == 1:
        faceN = int(line)
    elif len(line.split()) == 6:
        annotation = line.split()
        x=float(annotation[3])
        y=float(annotation[4])
        r=float(min(annotation[1],annotation[0]))
        print('%s x/y/r=%f/%f/%f'%(imagename,x,y,r))
    else:
        continue

