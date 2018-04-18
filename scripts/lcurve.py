#!/usr/bin/env python
import numpy as np
import math
import re
from pdb import *
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser(description='loss curve')
parser.add_argument('logf')
parser.add_argument('--plot',  '-p',    action='store_true')
parser.add_argument('--after', '-A', type=int, default=0)
args = parser.parse_args()

log_file = args.logf
with open(log_file) as f:
    l2 = f.readlines()

avg_min = 9999999
min_ech = 0
weights = min_weights = ""
lastavg = 0

points  = 0
graphY  = np.zeros(len(l2),dtype=np.float)

for m in l2:
    if str(m)[0]=='0':continue
    ech = re.search('[1-9][0-9]+',str(m))
    avg = re.findall('([0-9]+\.[0-9]+) +avg',str(m))
    wgt = re.findall('Saving weights to (\S+)',str(m))
    bup = len(re.findall('.backup',str(m)))
    if len(wgt)!=0 and bup != 1: weights = str(wgt[0])
    if len(avg)==0: continue
    if len(avg)==1:
        if args.after > int(ech.group()):continue
        points+=1
        avg = float(avg[0])
        graphY[points]=avg
        lastavg = avg
        if math.isnan(avg):lastavg = 9999999.
        epoch = int(ech.group())
        if avg_min>avg: avg_min, min_ech, min_weights = (avg, epoch, weights)
print('MinimalLoss: %d epoch %.5f loss %s'%(min_ech, avg_min, min_weights))
print('LastStage  : %d epoch %.5f loss %s'%(epoch, lastavg, weights))

try:
    if args.plot:
        GraphY  = np.zeros(points,dtype=np.float)
        GraphY  = graphY[0:points]
        print('points=%d loglen=%d'%(points,len(l2)))
        plt.plot(GraphY)
        plt.show()
except KeyboardInterrupt:
    print('stop')

