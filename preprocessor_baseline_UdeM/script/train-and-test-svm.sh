#!/bin/sh
set -e
PROJECTROOT=`dirname $0`/..
LIBLINEAR=$PROJECTROOT/lib/liblinear

# Parameters
# NUMLABELED: nb of labeled examples used to train the SVM
# NUMRUNS: numbers of train/test run (always the same test set)
NUMLABELED=100
NUMRUNS=50
C=1

# Train and test (logistic regression)
# $1: whole train set
# $2: whole test set
# $3: output file
$LIBLINEAR/run_all -s 0 -b 1 -c $C -l $NUMLABELED -r $NUMRUNS  -q $1 $2 $3

