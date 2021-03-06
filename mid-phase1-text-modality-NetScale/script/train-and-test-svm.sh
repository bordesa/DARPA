#!/bin/sh
set -e
PROJECTROOT=`dirname $0`/..
LIBLINEAR=$PROJECTROOT/src
LAUNCHDIR=`pwd`

# go to the script directory
cd $LIBLINEAR

# Train and test 
task=$1
# some care is taken to handle any path for the data (relative or absolute that is)
train_vec=$2
if [ ${#2} != 0 ]; then
    if [ ${2:0:1} != '/' ]; then  train_vec=$LAUNCHDIR/$2; fi
fi
train_lab=$3
if [ ${#3} != 0 ]; then
    if [ ${3:0:1} != '/' ]; then  train_lab=$LAUNCHDIR/$3; fi
fi
train_idx=$4
if [ ${#4} != 0 ]; then
    if [ ${4:0:1} != '/' ]; then  train_idx=$LAUNCHDIR/$4; fi
fi
test_vec=$5
if [ ${#5} != 0 ]; then
    if [ ${5:0:1} != '/' ]; then  test_vec=$LAUNCHDIR/$5; fi
fi
test_lab=$6
if [ ${#6} != 0 ]; then
    if [ ${6:0:1} != '/' ]; then  test_lab=$LAUNCHDIR/$6; fi
fi

python ClassifierSmartMemory.py $task $train_vec $train_lab $train_idx $test_vec $test_lab

# the prediction file is then brought back to the original directory
mv *.predictions $LAUNCHDIR/

# back to initial dir
cd $LAUNCHDIR

