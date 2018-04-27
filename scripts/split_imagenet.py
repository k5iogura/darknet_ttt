# -*- coding: utf-8 -*-

import os
import sys
import codecs

f = codecs.open(sys.argv[1], 'r', 'utf8', 'ignore')
lines = f.readlines()
id = ''
fl = None

for line in lines:
    line = line.strip()
    arr = line.split('\t')
    id_num = arr[0]
    url = arr[1]
    nid, num = id_num.split('_')
    
    if id != nid:
        id = nid
        if fl:
            fl.close()
        fl = codecs.open('lists/' + id + '.txt', 'a', 'utf8', 'ignore')
    fl.write(url + '\n')

fl.close()
f.close()
