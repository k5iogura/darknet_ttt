import argparse
import os
import sys
import re
import shutil
import fnmatch

parser = argparse.ArgumentParser(description='rename and copy image files for darknet classifier rule')
parser.add_argument('in_dir', nargs='+',  type=str, default='lfw/')
parser.add_argument('--execute',  '-e',   action='store_true', help='execute copy.')
parser.add_argument('--new_class',  '-c', type=str, default='person',help='person or automobile or etc.')
parser.add_argument('--rex',      '-r',   type=str, default='.jpg',help='jpg or (jpg|png) or etc.')
parser.add_argument('--append', '-a',     action='store_true',help='append new file, dont overwrite existed file')
parser.add_argument('--verbose', '-v',    action='store_true')
args = parser.parse_args()

if args.execute:
    print('execute copy files into directory')
else:
    print('not execute copy files into directory')

if len(args.in_dir)<2:
    print('need at least 2 directories on in_dir option')
    sys.exit(1)

in_dir = args.in_dir[:-1]
for in_d in in_dir:
    if not os.path.exists(in_d):
        print('%s directory not found'%in_d)
        sys.exit(1)

go_dir = args.in_dir[-1]
if go_dir in args.in_dir[:-1]:
    print('%s go directory is equal input directory')
    sys.exit(1)

if not os.path.exists(go_dir):
    print('Attention!:mkdir new %s directory'%go_dir)
    os.mkdir(go_dir)
elif not args.append:
    print('Attention!:output directory %s will be overrited'%go_dir)

files = []
rex = args.rex
for in_d in list(in_dir):
    for root,dirs,names in os.walk(in_d):
        finded_files = [os.path.join(root,f) for f in names
            if len(re.findall(rex,os.path.join(root,f)))>0 ]
        files.extend(finded_files)
    print('input directory %s\t%d files'%(in_d,len(files)))

new_class = args.new_class
no = 0
for f in files:
    if not os.path.isfile(f):continue
    sufix = os.path.splitext(f)
    if len(sufix) > 1:
        sufix = sufix[-1]
    new_file = os.path.join(go_dir,'%d_%s'%(no,'%s%s'%(new_class,sufix)))
    if args.append:
        while(True):
            if os.path.exists(new_file):
                no+=1
            else:
                break
            new_file = os.path.join(go_dir,'%d_%s'%(no,'%s%s'%(new_class,sufix)))
    if args.verbose: print('shutil.copy %s\t%s'%(f,new_file))
    if args.execute: shutil.copy(f,new_file)
    no+=1

print('*statistics*')
if not args.execute:print('This is simulation')
print('*directories input')
for d in in_dir:print('\t%s'%d)
print('*directories output')
ope=''
if args.append:ope='new '
print('\t%s directory %s%s class files %d'%(go_dir,ope,new_class,no))
sys.exit(1)

