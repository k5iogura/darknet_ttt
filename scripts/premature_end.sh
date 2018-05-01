#!/bin/bash

check_premature_end() {
    echo `od -t x2 $1 | tail -2 | grep d9ff | wc -l | awk '{if($1==0) print 1;else print 0}'`
}

for i in $*;
do
check_premature_end $i
done
