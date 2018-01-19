import os
import sys
import math
import re
import argparse
import scipy.io
import cv2
import numpy as np

def read_i(i_file):
    if os.path.exists(i_file):
        image=cv2.imread(i_file)
        return image
    return None

def view_i(image):
    cv2.imshow('image-check',image)
    while True:
        k=cv2.waitKey(100)
        if k==1048603: sys.exit(1)#ESC
        elif k==1048608: break    #space?
        if k==27: sys.exit(1)     #ESC
        elif k==32: break         #space?
        elif k!=-1: print(k)

def convert(size, box):
    #The Same as YOLO Format
    # classid <center-x> <center-y> <width> <height>
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

parser = argparse.ArgumentParser()
parser.add_argument('mat',type=str)
parser.add_argument('--view', '-v', action='store_true')
parser.add_argument('--list_prefix', '-p', type=str, default='train')
parser.add_argument('--shuffle', '-sh', action='store_true')
parser.add_argument('--start', '-s', type=int, default=0)
args = parser.parse_args()

mat_key = os.path.basename(args.mat)
mat_key = re.sub('.mat','',str(mat_key))
train_txt = args.list_prefix+'_'+mat_key+'.txt'
print('mat_key=%s train text=%s'%(mat_key,train_txt))
meta = scipy.io.loadmat(args.mat)
if not mat_key in meta.keys():
    print('%s not found in mat file'%(mat_key))
    print(meta.keys())
    sys.exit(1)
full_paths     = meta[mat_key][0,0]['full_path'][0]
face_locations = meta[mat_key][0,0]['face_location'][0]
face_scores    = meta[mat_key][0,0]['face_score'][0]
face_scores_2nd= meta[mat_key][0,0]['second_face_score'][0]

dataN = len(full_paths)
wdir  = str(os.getcwd())+'/'+os.path.dirname(args.mat)+'/'

print('image data = %d'%(dataN))
validN=0
index = range(0,dataN)
if args.shuffle is True:
    index = np.random.permutation(dataN)
    full_paths     =full_paths[index]
    face_locations =face_locations[index]
    face_scores    =face_scores[index]
    face_scores_2nd=face_scores_2nd[index]

lost_file=None
if args.view is False:
    itxt = open(train_txt,'w')
for i in range(args.start,dataN):
    i_file  = wdir+str(full_paths[i][0])
    t_file  = re.sub('.jpg','.txt',i_file)
    t_file  = re.sub('.png','.txt',t_file)
    f_loc   = face_locations[i][0]
    score   = face_scores[i]
    score2nd= face_scores_2nd[i]
    if not (score <10. and score >0.): continue
    if math.isnan(score):continue
    if math.isnan(score2nd) is False:continue

    if not os.path.exists(i_file):
        if lost_file is None: lost_file = open('LOST_FILE.txt','a')
        lost_file.write('%s\n'%(i_file))
        print('not found   %s'%(i_file))
        lost_file.close()
        lost_file = None
        continue
    image=read_i(i_file)
    if image is None:
        if lost_file is None: lost_file = open('LOST_FILE.txt','a')
        lost_file.write('%s\n'%(i_file))
        print('cannot read %s'%(i_file))
        lost_file.close()
        lost_file = None
        continue
    h,w,c = image.shape
    lx,ly,rx,ry = f_loc.astype(dtype=np.int)
    if h < ry or w < rx: continue
    if h > w+5: continue
    #if h <= w: continue

    cx,cy,aw,ah = convert((int(image.shape[1]),int(image.shape[0])),(lx,rx,ly,ry))
    if args.view is False:
        itxt.write('%s\n'%(i_file))
        with open(t_file,'w') as ltxt:
            ltxt.write('14 %f %f %f %f\n'%(cx,cy,aw,ah))

    validN+=1
    if validN%1000==0:print('\tchecked %d...'%(validN))
    if args.view:
        print('%s'%(i_file))
        print('%s'%(t_file))
        print('14 %f %f %f %f'%(cx,cy,aw,ah))
        print('image',image.shape)
        print(score)
        lx,ly,rx,ry=f_loc.astype(dtype=np.int)
        cv2.rectangle(image,(lx,ly),(rx,ry),(0,0,255),2)
        view_i(image)

if args.view is False:
    itxt.close()
else:
    cv2.destroyAllWindows()
print('images valid/total= %d/%d'%(validN,dataN))

