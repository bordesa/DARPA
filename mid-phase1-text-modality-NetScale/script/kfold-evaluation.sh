#!/bin/sh
set -e
LAUNCHDIR=`pwd`
cd `dirname $0`
PROJECTROOT=..
DATADIR=$PROJECTROOT

export PYTHONPATH=$PROJECTROOT/DLmodel:$PYTHONPATH

KFOLD=10
TRAINSIZE=10000
SEED=777

THEANO_FLAGS=device=gpu,floatX=float32 python $PROJECTROOT/src/DARPAEvaluation.py $KFOLD $TRAINSIZE $PROJECTROOT/preprocessed-opentable $PROJECTROOT/DLmodel/Saved_Model/ $SEED

cd $LAUNCHDIR

