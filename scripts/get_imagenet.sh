#!/bin/bash

#enable job control
set -m

#download and cleanup sub-shell
get_1list() {
    export dir=`basename $1 | sed 's/\.txt//'`
    if [ -d images/$dir ];
    then
        echo images/$dir already found, skipped >> images/${dir}.log
        return
    fi

    echo downloading $dir
    wget -i $1 -P images/$dir -T 1 -t 1 -nc >& /dev/null
    echo wget done >> images/${dir}.log

    if [ ! -d images/$dir ];
    then
        echo images/$dir not found
        return
    fi

    echo cleaning $dir >> images/${dir}.log
    for j in `find images/$dir -name \*.[jJ]\*`;
    do
        export JPEG=`file $j | grep JPEG | wc -l | awk '{print $1;}'`
        if [ $JPEG -eq 0 ];
        then
            rm -f $j
        fi
    done
    echo clean up $dir done >> images/${dir}.log
}

#main
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
for i in `find $1 -name \*.txt`;
do
    get_1list $i &
done

#wait sub-shell exiting
#fg
