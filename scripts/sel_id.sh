#!/bin/bash
for i in `find VOCdevkit/VOC2007/JPEGImages/ -name \*.txt|sort -R`;
do
export i14=`grep '^14' $i | wc -l`
export i08=`grep '^8'  $i | wc -l`
export all=`wc -l $i|awk '{print $1;}'`
export oth=`expr $all - $i14 - $i08`

if [ $i14 -ne 0 ] || [ $i08 -ne 0 ] && [ $oth -eq 0 ];
then
echo `pwd`/$i | sed 's/\.txt/.jpg/'
fi
done
