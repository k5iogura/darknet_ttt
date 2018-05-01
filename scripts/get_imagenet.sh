#!/bin/bash

#disable job control
#set -m

#checker for jpg
check_premature_end() {
    echo `od -t x2 $1 | tail -2 | grep d9ff | wc -l | awk '{if($1==0) print 1;else print 0}'`
}

#download and cleanup sub-shell
get_1list() {
    dir=`basename $1 | sed 's/\.txt//'`
    if [ -d images/$dir ];
    then
#        echo images/$dir already found, skipped >> images/${dir}.log
        return
    fi

    echo downloading $dir
    wget -i $1 -P images/$dir -T 1 -t 1 -nc >& /dev/null
    echo wget done >> images/${dir}.log

    if [ ! -d images/$dir ];
    then
        echo images/$dir dont retrieved >> images/lost-images.log
        return
    fi

    echo cleaning $dir >> images/${dir}.log
    for j in `find images/$dir -name \*.[jJ]\*`;
    do
        PEND=`check_premature_end $j`
        if [ $PEND -eq 1 ];
        then
            rm -f $j
        fi
        JPEG=`file $j | grep JPEG | wc -l | awk '{print $1;}'`
        if [ $JPEG -eq 0 ];
        then
            rm -f $j
        fi
    done
    echo clean up $dir done >> images/${dir}.log
}

#main
max_launch=50
if [ $# -ne 1 ];
then
    echo Usage $0 lists
    exit
fi

if [ ! -d $1 ];
then
    echo $1 directory not found
    exit
fi

#execute sub-shell with multi-process
c=0
for i in `find $1 -name \*.txt`;
do
    (get_1list $i) &
    c=`expr $c + 1`
    if [ $c -eq $max_launch ];
    then
        echo wait $c sub-shell
        wait
        c=0
    fi
done

