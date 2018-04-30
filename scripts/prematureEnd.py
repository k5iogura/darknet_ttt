#!/usr/bin/env python
#coding:utf-8
import cv2
import sys
import re
import os
import codecs

filename = sys.argv[1]
if len(re.findall('.txt',filename))>0:
    print 'txt file'
    with codecs.open(filename,'r','utf-8','ignore') as f:
        cont = f.read().strip().split('\n')
else:
    cont = [filename]

for i in cont:
    if os.path.exists(i) is False:
        print('%s'%(i))
        continue
    with open(i, 'rb') as f:
        check_chars = f.read()[-2:]
    if check_chars != b'\xff\xd9':
        print('%s'%(i))
        #print('%s\tNot complete image'%(i))
        #img = cv2.imread(i)
#    else:
#        img = cv2.imread(i)
