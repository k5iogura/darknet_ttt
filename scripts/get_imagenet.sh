#!/bin/bash

for i in `find lists -name \*.txt`;
do
    export dir=`basename $i | sed 's/\.txt//'`
    echo $dir
    wget -i $i -P images/$dir -T 1 -t 1 -nc >& log

echo clean up $dir
for j in `find images/$dir`;
do
export JPEG=`file $j | grep JPEG | wc -l | awk '{print $1;}'`
if [ $JPEG -eq 0 ];
then
    rm -f $j
fi
done
done
