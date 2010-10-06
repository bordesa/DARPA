#!/bin/sh
set -e
PROJECTROOT=`dirname $0`/..
LIBLINEAR=$PROJECTROOT/lib/liblinear
DATADIR=$PROJECTROOT
DBDIR=/path/to/your/openTable/dbfiles

# Generate the data (preprocessing and reducing the dictionary size to DICTSIZE)
# and split it according to the index given in TrainRestaurantsIds-exemple.dat
# Note if DICTSIZE=0, the dictionary contains all the words
DICTSIZE=5000
python $PROJECTROOT/src/dbToBoW_DARPA.py $DATADIR/featDict.txt $DATADIR/preprocessed-opentable $PROJECTROOT/script/TrainRestaurantsIds-example.dat $DICTSIZE $DBDIR/opentable.*.db



