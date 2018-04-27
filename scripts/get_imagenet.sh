#!/bin/bash

export getn=0
export rmvn=0

for i in `find lists -name \*.txt`;
do
    export dir=`basename $i | sed 's/\.txt//'`
    echo downloading $dir
    wget -i $i -P images/$dir -T 1 -t 1 -nc >& log

echo clean up $dir
if [ ! -d $i ];
then
continue
fi
for j in `find images/$dir -name \*.[jJ]\*`;
do
export JPEG=`file $j | grep JPEG | wc -l | awk '{print $1;}'`
if [ $JPEG -eq 0 ];
then
    rm -f $j
    export rmvn=`expr $rmvn + 1`
else
    export getn=`expr $getn + 1`
fi
done
echo getn/rmvn = $getn / $rmvn
done
