#!/bin/bash

cd test
rm -rf test.txt
touch test.txt

TOP=$(ls *.jl)
for f in $TOP
do
    if [ $f != "runtests.jl" ]
    then
        echo $f >> test.txt
    fi
done

PROBLEM=$(ls problems/*.jl)
for f in $PROBLEM
do
    echo $f >> test.txt
done

ROUTINES=$(ls optimization_routines/*.jl)
for f in $ROUTINES
do
    echo $f >> test.txt
done