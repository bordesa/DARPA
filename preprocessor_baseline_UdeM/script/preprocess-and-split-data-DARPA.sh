#!/bin/sh
set -e
PROJECTROOT=`dirname $0`/..
LIBLINEAR=$PROJECTROOT/lib/liblinear
DATADIR=$PROJECTROOT
DBDIR=/data/lisa5/DARPA/OpenTable/db/

# Generate the data (preprocessing and reducing the dictionary size to DICTSIZE)
# Note if DICTSIZE=0, the dictionary contains all the words
DICTSIZE=5000
python $PROJECTROOT/src/dbToBoW_numpy.py $DATADIR/testDict.txt $DATADIR/testData.feat $DATADIR/testLabel.lab $DICTSIZE $DBDIR/opentable.*.db
# IMPORTANT COMMENT:
# if you don't want to directly generate the libsvm file but rather
# want to write the text of reviews (after preprocessing and filtering
# to DICTSIZE) comment the line above and uncomment the one below.
#python $PROJECTROOT/src/dbToText.py $DATADIR/testDict.txt $DATADIR/testData.feat $DICTSIZE $DBDIR/opentable.*.db

# Split test and training set
# Note: The following split corresponds to our current split at UdeM
NUMTRAIN=373532
NUMTEST=62254
head -n $NUMTRAIN $DATADIR/testData.feat > $DATADIR/train.vect
tail -n $NUMTEST  $DATADIR/testData.feat > $DATADIR/test.vect
head -n $NUMTRAIN $DATADIR/testLabel.lab > $DATADIR/train.lab
tail -n $NUMTEST  $DATADIR/testLabel.lab > $DATADIR/test.lab
