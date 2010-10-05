#!/bin/sh
set -e
PROJECTROOT=`dirname $0`/..
LIBLINEAR=$PROJECTROOT/lib/liblinear
DATADIR=$PROJECTROOT
DBDIR=/data/lisa5/DARPA/OpenTable/db

# Generate the data (preprocessing and reducing the dictionary size to DICTSIZE)
# and split it according to the index given in myTrainRestaurantsIds.dat
# Note if DICTSIZE=0, the dictionary contains all the words
DICTSIZE=5000
python $PROJECTROOT/src/dbToBoW_DARPA.py $DATADIR/featDict.txt $DATADIR/preprocessed-opentable myTrainRestaurantsIds.dat $DICTSIZE $DBDIR/opentable.*.db



