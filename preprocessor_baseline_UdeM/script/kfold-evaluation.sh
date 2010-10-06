#!/bin/sh
set -e
PROJECTROOT=`dirname $0`/..
DATADIR=$PROJECTROOT

export PYTHONPATH=$PROJECTROOT/DLmodel:$PYTHONPATH

KFOLD=10
TRAINSIZE=10000
SEED=777

THEANO_FLAGS=device=gpu,floatX=float32 python $PROJECTROOT/src/DARPAEvaluation.py $KFOLD $TRAINSIZE $PROJECTROOT/preprocessed-opentable $PROJECTROOT/DLmodel/Saved_Model/ $SEED



